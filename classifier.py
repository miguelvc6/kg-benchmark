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

import ijson

QID_RE = re.compile(r"^Q[1-9][0-9]*$")
PID_RE = re.compile(r"^P[1-9][0-9]*$")
DATE_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Default paths
DEFAULT_REPAIRS_PATH = "02_wikidata_repairs.json"
DEFAULT_WORLD_STATE_PATH = "03_world_state.json"
DEFAULT_POPULARITY_PATH = "00_entity_popularity.json"
DEFAULT_OUT_PATH = "04_classified_benchmark.jsonl"
DEFAULT_OUT_FULL_PATH = "04_classified_benchmark_full.jsonl"
DEFAULT_STATS_PATH = "reports/classifier_stats.json"

# These are the only ones we *confidently* treat as "Logical" without local/external info.
# Note: we intentionally key off the report violation type (Stage 1/2), not "all constraints on property".
LOGICAL_REPORT_TYPES = {
    "format",
    "one of",
    "range",
}

# Optional: map report type -> canonical constraint type QID (when you want it in outputs)
REPORT_TYPE_TO_CONSTRAINT_QID = {
    "format": "Q21502404",  # format constraint
    "one of": "Q21502402",  # one-of constraint
    "range": "Q21510860",  # range constraint
}


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def is_qid(x: Any) -> bool:
    return isinstance(x, str) and bool(QID_RE.match(x))


def is_pid(x: Any) -> bool:
    return isinstance(x, str) and bool(PID_RE.match(x))


def normalize_text(s: str) -> str:
    s = s.lower()
    # keep digits/letters; collapse whitespace
    s = re.sub(r"[^\w\s\-:/\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_repairs(path: Path) -> Iterator[Dict[str, Any]]:
    """
    Supports:
      - .jsonl: one object per line
      - .json: either a JSON array, or a single JSON object (not expected here)
    For large arrays, uses ijson if available.
    """
    if path.suffix == ".jsonl":
        yield from iter_jsonl(path)
        return

    # Peek first non-whitespace char
    with open(path, "r", encoding="utf-8") as fh:
        start = fh.read(2048)
    first = next((c for c in start if not c.isspace()), "")

    if first == "[":
        if ijson is None:
            data = read_json(path)
            if not isinstance(data, list):
                raise ValueError(f"Expected a list in {path}, got {type(data)}")
            for obj in data:
                if not isinstance(obj, dict):
                    continue
                yield obj
        else:
            with open(path, "r", encoding="utf-8") as fh:
                for obj in ijson.items(fh, "item"):
                    if isinstance(obj, dict):
                        yield obj
        return

    # Fallback: assume a single object (rare/unexpected)
    obj = read_json(path)
    if isinstance(obj, dict):
        yield obj
    elif isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                yield x
    else:
        raise ValueError(f"Unsupported JSON content in {path}: {type(obj)}")


def safe_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


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


def get_truth_tokens(repair_event: Dict[str, Any]) -> List[str]:
    """
    For A-BOX UPDATE/CREATE: truth is the post-repair value.
    Falls back to persistence / violation context if missing.
    For DELETE and T-BOX: no truth; classifier treats separately.
    """
    rt = repair_event.get("repair_target", {})
    if not isinstance(rt, dict):
        return []

    kind = rt.get("kind")
    if kind != "A_BOX":
        return []

    action = rt.get("action")
    if action == "DELETE":
        return []

    candidates = [
        rt.get("new_value"),
        rt.get("value"),
        safe_get(repair_event, "persistence_check", "current_value_2025"),
        safe_get(repair_event, "violation_context", "value_current_2025"),
    ]
    for v in candidates:
        toks = flatten_truth(v)
        if toks:
            return toks
    return []


def _pre_repair_value(repair_event: Dict[str, Any]) -> Tuple[Any, str]:
    rt = repair_event.get("repair_target", {})
    if isinstance(rt, dict) and "old_value" in rt:
        return rt.get("old_value"), "repair_target.old_value"
    vc_val = safe_get(repair_event, "violation_context", "value")
    if vc_val is not None:
        return vc_val, "violation_context.value"
    return None, "missing"


def local_context_ids_and_texts(
    repair_event: Dict[str, Any],
    world_state_entry: Dict[str, Any],
) -> Tuple[set, str, Dict[str, Any]]:
    """
    Builds (local_ids_set, local_text_blob) from:
      - focus node label/description (Stage2 + fallback to L1)
      - 1-hop outgoing edges (L3), with synthetic pre-repair values
        for the target property to avoid post-repair leakage.
    """
    local_ids: set = set()
    texts: List[str] = []
    synth_info = {"used_pre_repair_value": False, "pre_repair_source": "missing", "tokens": []}

    # Focus node text
    q_label = repair_event.get("qid_label_en") or safe_get(world_state_entry, "L1_ego_node", "label")
    q_desc = repair_event.get("qid_description_en") or safe_get(world_state_entry, "L1_ego_node", "description")
    for x in (q_label, q_desc):
        if isinstance(x, str) and x.strip():
            texts.append(x)

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
                local_ids.add(tq)
            tl = e.get("target_label")
            td = e.get("target_description")
            if isinstance(tl, str) and tl.strip():
                texts.append(tl)
            if isinstance(td, str) and td.strip():
                texts.append(td)

    # Also include constraint-readable labels? (optional; excluded to keep local vs logical separation clean)

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
                    local_ids.add(tok)
                else:
                    texts.append(str(tok))
        else:
            synth_info["pre_repair_source"] = pre_src
    else:
        synth_info["pre_repair_source"] = "not_applicable"

    blob = normalize_text(" \n ".join(texts))
    return local_ids, blob, synth_info


def report_type_normalized(repair_event: Dict[str, Any]) -> str:
    vc = repair_event.get("violation_context", {})
    if not isinstance(vc, dict):
        return ""
    t = vc.get("report_violation_type_normalized") or vc.get("report_violation_type") or ""
    if not isinstance(t, str):
        return ""
    return normalize_text(t)


def is_logical_violation(repair_event: Dict[str, Any]) -> bool:
    t = report_type_normalized(repair_event)
    # Some report types include extra words; use containment heuristics carefully.
    # We only treat as logical if it matches the canonical names.
    for k in LOGICAL_REPORT_TYPES:
        if t == k:
            return True
    return False


def logical_from_constraints(constraint_types: List[Dict[str, Optional[str]]]) -> bool:
    for c in constraint_types:
        if not isinstance(c, dict):
            continue
        qid = c.get("qid")
        label = c.get("label_en")
        if qid in REPORT_TYPE_TO_CONSTRAINT_QID.values():
            return True
        if isinstance(label, str):
            ln = normalize_text(label)
            if ln in {"format constraint", "one of constraint", "one-of constraint", "range constraint"}:
                return True
            if ln in {"format", "one of", "one-of", "range"}:
                return True
    return False


def match_truth_locally(truth_tokens: List[str], local_ids: set, local_text: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Returns:
      - matched: bool
      - evidence: dict with counts and examples
    Matching policy:
      - QIDs: must appear in local_ids
      - ISO dates: exact substring
      - Other literals: exact substring on normalized text
    """
    if not truth_tokens:
        return False, {"needed": 0, "found": 0, "matches": []}

    needed = 0
    found = 0
    matches: List[Dict[str, Any]] = []
    used_literal_substring = False

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
            ok = tok_s in local_ids
            if ok:
                found += 1
                matches.append({"token": tok_s, "kind": "qid", "where": "L3_neighborhood"})
            continue

        # literal/date matching on normalized text
        if DATE_ISO_RE.match(tok_s):
            ok = normalize_text(tok_s) in local_text  # exact date required
            if ok:
                found += 1
                matches.append({"token": tok_s, "kind": "date", "where": "text"})
            continue

        tok_n = normalize_text(tok_s)
        ok = tok_n and tok_n in local_text
        if ok:
            found += 1
            used_literal_substring = True
            matches.append({"token": tok_s, "kind": "literal", "where": "text"})

    matched = needed > 0 and found == needed
    evidence = {
        "needed": needed,
        "found": found,
        "matches": matches,
        "used_literal_substring": used_literal_substring,
        "local_ids_count": len(local_ids),
    }
    return matched, evidence


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

    if not isinstance(world_state_entry, dict):
        # no context => we cannot safely do local check; treat as external with low confidence
        trace = [
            {"step": "is_delete", "result": None},
            {"step": "local_availability", "result": "missing_world_state"},
            {"step": "logical_whitelist", "result": is_logical_violation(repair_event)},
            {"step": "fallback_external", "result": True},
            {"step": "branch", "result": "missing_world_state"},
        ]
        return (
            make("TypeC", "EXTERNAL", "low", trace, "Missing world_state entry; defaulted to external.", []),
            "missing_world_state",
            {"missing_truth_tokens": True, "missing_old_value": False},
        )

    constraint_types = extract_constraint_types(world_state_entry)

    track = repair_event.get("track")
    rt = repair_event.get("repair_target", {}) if isinstance(repair_event.get("repair_target"), dict) else {}

    # T-BOX: taxonomy does not strictly apply (different task: rule drift detection).
    if track == "T_BOX" or rt.get("kind") == "T_BOX":
        trace = [
            {"step": "is_delete", "result": False},
            {"step": "local_availability", "result": "not_applicable_t_box"},
            {"step": "logical_whitelist", "result": None},
            {"step": "fallback_external", "result": None},
            {"step": "branch", "result": "t_box_unknown"},
        ]
        return (
            make(
                "UNKNOWN",
                "UNKNOWN",
                "low",
                trace,
                "T-Box (Reformer) case: information-necessity taxonomy is defined for A-Box value repairs; labeled UNKNOWN.",
                constraint_types,
            ),
            None,
            {"missing_truth_tokens": True, "missing_old_value": False},
        )

    # A-BOX delete => Type A rejection
    action = rt.get("action")
    if action == "DELETE":
        trace = [
            {"step": "is_delete", "result": True},
            {"step": "local_availability", "result": False},
            {"step": "logical_whitelist", "result": False},
            {"step": "fallback_external", "result": False},
            {"step": "branch", "result": "delete_rejection"},
        ]
        return (
            make(
                "TypeA",
                "REJECTION",
                "high",
                trace,
                "A-Box DELETE action treated as logical rejection/cleaning.",
                constraint_types,
            ),
            None,
            {"missing_truth_tokens": True, "missing_old_value": False},
        )

    truth_tokens = get_truth_tokens(repair_event)
    missing_truth = len(truth_tokens) == 0

    local_ids, local_text, synth_info = local_context_ids_and_texts(repair_event, world_state_entry)
    matched, evidence = match_truth_locally(truth_tokens, local_ids, local_text)

    if matched:
        conf = "high"
        if evidence.get("used_literal_substring"):
            conf = "medium"
        if isinstance(evidence.get("needed"), int) and evidence["needed"] > 3:
            conf = "medium"
        trace = [
            {"step": "is_delete", "result": False},
            {"step": "local_availability", "result": True, "evidence": evidence, "synthetic": synth_info},
        ]
        # still include the remaining steps for debuggability
        trace.append({"step": "logical_whitelist", "result": False})
        trace.append({"step": "fallback_external", "result": False})
        trace.append({"step": "branch", "result": "local_match"})
        return (
            make(
                "TypeB",
                "LOCAL",
                conf,
                trace,
                "Truth tokens found in local context using synthetic pre-repair values for target property.",
                constraint_types,
            ),
            None,
            {"missing_truth_tokens": False, "missing_old_value": synth_info.get("pre_repair_source") == "missing"},
        )

    # Logical whitelist check based on report violation type
    logical = False
    logical_signal = "report_type"
    if constraint_types:
        logical = logical_from_constraints(constraint_types)
        logical_signal = "L4_constraints"
    else:
        logical = is_logical_violation(repair_event)
    if logical:
        trace = [
            {"step": "is_delete", "result": False},
            {"step": "local_availability", "result": False, "evidence": evidence, "synthetic": synth_info},
            {
                "step": "logical_whitelist",
                "result": True,
                "signal": logical_signal,
                "report_type": report_type_normalized(repair_event),
            },
            {"step": "fallback_external", "result": False},
            {"step": "branch", "result": "logical_constraint"},
        ]
        return (
            make(
                "TypeA",
                "LOGICAL",
                "high",
                trace,
                "Violation type indicates a logical constraint solvable from the rule definition.",
                constraint_types,
            ),
            None,
            {
                "missing_truth_tokens": missing_truth,
                "missing_old_value": synth_info.get("pre_repair_source") == "missing",
            },
        )

    # Default: external
    conf = "medium"
    if missing_truth:
        conf = "low"
    trace = [
        {"step": "is_delete", "result": False},
        {"step": "local_availability", "result": False, "evidence": evidence, "synthetic": synth_info},
        {
            "step": "logical_whitelist",
            "result": False,
            "signal": logical_signal if constraint_types else "report_type",
            "report_type": report_type_normalized(repair_event),
        },
        {"step": "fallback_external", "result": True},
        {"step": "branch", "result": "external_fallback"},
    ]
    return (
        make(
            "TypeC",
            "EXTERNAL",
            conf,
            trace,
            "Truth not found locally (using synthetic pre-repair target property) and violation is not purely logical; treated as external.",
            constraint_types,
        ),
        None,
        {
            "missing_truth_tokens": missing_truth,
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
    local_ids, local_text, synth_info = local_context_ids_and_texts(repair_event, world_state_entry)
    assert "Q99" not in local_ids, "new_value leaked into local_ids"
    assert "Q1" in local_ids, "old_value missing from synthetic local_ids"
    matched, _ = match_truth_locally(["Q99"], local_ids, local_text)
    assert not matched, "new_value should not match local context via target property"
    assert synth_info["used_pre_repair_value"], "synthetic pre-repair value not used"

    # ISO date precision: full date must match exactly.
    local_text = normalize_text("born 1999-01-02")
    matched, _ = match_truth_locally(["1999-01-02"], set(), local_text)
    assert matched, "ISO date should match exact full date"
    matched, _ = match_truth_locally(["1999-01-03"], set(), local_text)
    assert not matched, "ISO date should not match a different date"


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
        for repair_event in iter_repairs(repairs_path):
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
