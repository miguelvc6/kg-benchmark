from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_SELECTION_SEED = 13
DEFAULT_SELECTED_CASE_ORDER = "sorted"
SUPPORTED_SELECTED_CASE_ORDERS = {"sorted", "shuffled"}
TRACK_TBOX_MARKER = '"track": "T_BOX"'
CASE_ID_RE = re.compile(r'"id"\s*:\s*"([^"]+)"')
PROPERTY_REVISION_RE = re.compile(r'"property_revision_id"\s*:\s*"?([^",}\s]+)"?')
WIKIDATA_DATE_RE = re.compile(r"^[+-]\d{4,}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}Z)?$")
DEFAULT_CORE_SIZE = 4800
DEFAULT_DEV_SIZE = 600
DEFAULT_TBOX_CAP_CORE = 10
DEFAULT_TBOX_CAP_DEV = 3
DEFAULT_TBOX_CAP_PER_UPDATE = 100
DEFAULT_ABOX_CAP_CORE = 3
DEFAULT_ABOX_CAP_DEV = 2

CORE_QUOTAS: dict[str, int] = {
    "TypeA_REJECTION_FORMAT_INVALID": 200,
    "TypeA_FORMAT_NORMALIZATION": 160,
    "TypeA_FORMAT_VALUE_PRUNING": 120,
    "TypeA_SELF_LINK_REJECTION": 80,
    "TypeA_SET_MEMBERSHIP_REJECTION": 40,
    "TypeA_MULTIPLICITY_NORMALIZATION": 40,
    "TypeA_TARGET_REQUIRED_CLAIM": 40,
    "TypeA_REJECTION_RULE_INVALID": 20,
    "TypeA_LOGICAL": 40,
    "TypeA_DELETE_AMBIGUOUS": 250,
    "TypeB_LOCAL_TEXT_CONFIRMED": 520,
    "TypeB_LOCAL_TEXT_DERIVED": 100,
    "TypeB_LOCAL_SELECTION_CONFIRMED": 420,
    "TypeB_LOCAL_FOCUS_QID": 80,
    "TypeB_LOCAL_FOCUS_NON_TARGET_PROPERTY": 70,
    "TypeB_LOCAL_MIXED": 38,
    "TypeB_LOCAL_NEIGHBOR_IDS": 2,
    "TypeC_EXTERNAL_BY_ELIMINATION": 900,
    "TBOX_RELAXATION_SET_EXPANSION": 650,
    "TBOX_RESTRICTION_SET_CONTRACTION": 250,
    "TBOX_SCHEMA_UPDATE": 600,
    "TBOX_COINCIDENTAL_SCHEMA_CHANGE": 300,
}

DEV_QUOTAS: dict[str, int] = {
    "DEV_TYPEA_CLEAN_RULE_REJECTION": 70,
    "DEV_TYPEA_AMBIGUOUS_DELETE": 40,
    "DEV_TYPEB_LOCAL": 130,
    "DEV_TYPEC_EXTERNAL_OR_UNKNOWN": 120,
    "DEV_TBOX_RELAXATION_SET_EXPANSION": 80,
    "DEV_TBOX_RESTRICTION_SET_CONTRACTION": 40,
    "DEV_TBOX_SCHEMA_UPDATE": 80,
    "DEV_TBOX_COINCIDENTAL_SCHEMA_CHANGE": 40,
}

CORE_BACKFILL: dict[str, list[str]] = {
    "TypeA_REJECTION_RULE_INVALID": ["TypeA_REJECTION_FORMAT_INVALID"],
    "TypeA_LOGICAL": ["TypeA_REJECTION_FORMAT_INVALID", "TypeA_FORMAT_NORMALIZATION"],
    "TypeA_SET_MEMBERSHIP_REJECTION": ["TypeA_SELF_LINK_REJECTION", "TypeA_FORMAT_VALUE_PRUNING"],
    "TypeA_MULTIPLICITY_NORMALIZATION": ["TypeA_FORMAT_NORMALIZATION"],
    "TypeB_LOCAL_NEIGHBOR_IDS": ["TypeB_LOCAL_TEXT_CONFIRMED", "TypeB_LOCAL_SELECTION_CONFIRMED"],
    "TypeB_LOCAL_MIXED": ["TypeB_LOCAL_TEXT_CONFIRMED", "TypeB_LOCAL_SELECTION_CONFIRMED"],
    "TypeB_LOCAL_FOCUS_NON_TARGET_PROPERTY": ["TypeB_LOCAL_TEXT_CONFIRMED", "TypeB_LOCAL_SELECTION_CONFIRMED"],
    "TypeB_LOCAL_TEXT_DERIVED": ["TypeB_LOCAL_TEXT_CONFIRMED"],
    "TypeB_LOCAL_FOCUS_QID": ["TypeB_LOCAL_SELECTION_CONFIRMED", "TypeB_LOCAL_TEXT_CONFIRMED"],
    "TBOX_RESTRICTION_SET_CONTRACTION": ["TBOX_RELAXATION_SET_EXPANSION", "TBOX_SCHEMA_UPDATE"],
}

DIAGNOSTIC_SUBTYPES = {
    "DELETE_AMBIGUOUS",
    "COINCIDENTAL_SCHEMA_CHANGE",
    "UNKNOWN_SELECTION_AMBIGUOUS",
    "UNKNOWN_MULTIPLICITY_ARTIFACT",
    "UNKNOWN_INCOMPLETE_LOCAL_CONTEXT",
    "UNKNOWN_MISSING_TRUTH",
    "UNKNOWN_MISSING_WORLD_STATE",
    "UNKNOWN_CURRENT_VALUE_FALLBACK",
    "UNKNOWN_FORMAT_PRUNING_RETAINED_UNVERIFIED",
    "UNKNOWN_BAD_TARGET_OR_CONTEXT",
    "UNKNOWN_FOCUS_QID_DOMAIN_REASONING",
    "UNKNOWN_TBOX_CAUSALITY",
}
MAIN_SCORE_STRATA = {
    "TypeA_REJECTION_FORMAT_INVALID",
    "TypeA_FORMAT_NORMALIZATION",
    "TypeA_FORMAT_VALUE_PRUNING",
    "TypeA_SELF_LINK_REJECTION",
    "TypeA_SET_MEMBERSHIP_REJECTION",
    "TypeA_MULTIPLICITY_NORMALIZATION",
    "TypeA_TARGET_REQUIRED_CLAIM",
    "TypeA_REJECTION_RULE_INVALID",
    "TypeA_LOGICAL",
    "TypeB_LOCAL_TEXT_CONFIRMED",
    "TypeB_LOCAL_TEXT_DERIVED",
    "TypeB_LOCAL_SELECTION_CONFIRMED",
    "TypeB_LOCAL_FOCUS_QID",
    "TypeB_LOCAL_FOCUS_NON_TARGET_PROPERTY",
    "TypeB_LOCAL_MIXED",
    "TypeB_LOCAL_NEIGHBOR_IDS",
    "TypeC_EXTERNAL_BY_ELIMINATION",
    "TBOX_RELAXATION_SET_EXPANSION",
    "TBOX_RESTRICTION_SET_CONTRACTION",
    "TBOX_SCHEMA_UPDATE",
}


@dataclass(frozen=True)
class SelectionOptions:
    classified_benchmark: Path
    tier: str
    seed: int = DEFAULT_SELECTION_SEED
    core_size: int = DEFAULT_CORE_SIZE
    dev_size: int = DEFAULT_DEV_SIZE
    tbox_cap_core: int = DEFAULT_TBOX_CAP_CORE
    tbox_cap_dev: int = DEFAULT_TBOX_CAP_DEV
    abox_cap_core: int = DEFAULT_ABOX_CAP_CORE
    abox_cap_dev: int = DEFAULT_ABOX_CAP_DEV
    selected_case_order: str = DEFAULT_SELECTED_CASE_ORDER
    progress_every: int = 100000
    exclude_manifest: Path | None = None
    quotas: dict[str, int] | None = None


def _ordered_unique_case_ids(case_ids: Optional[Iterable[str]]) -> list[str]:
    if not case_ids:
        return []
    ordered: list[str] = []
    seen: set[str] = set()
    for case_id in case_ids:
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            continue
        seen.add(case_id)
        ordered.append(case_id)
    return ordered


def _stable_digest(seed: int, tier: str, selection_stratum: str, group_key: str, case_id: str) -> str:
    return hashlib.sha1(f"{seed}|{tier}|{selection_stratum}|{group_key}|{case_id}".encode()).hexdigest()


def _selected_case_sort_key(
    case_id: str,
    annotations: dict[str, dict[str, Any]],
    *,
    seed: int,
    tier: str,
    selected_case_order: str,
) -> tuple[str, str] | str:
    if selected_case_order == "sorted":
        return case_id
    ann = annotations[case_id]
    return (
        _stable_digest(seed, tier, ann["selection_stratum"], ann["group_key"], case_id),
        case_id,
    )


def _legacy_selected_case_sort_key(case_id: str, seed: int, selected_case_order: str) -> tuple[str, str] | str:
    if selected_case_order == "sorted":
        return case_id
    payload = f"{seed}|selected_case_ids|{case_id}".encode("utf-8")
    return (hashlib.sha1(payload).hexdigest(), case_id)


def _legacy_tbox_rank(case_id: str, property_revision_id: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}|{property_revision_id}|{case_id}".encode("utf-8")).hexdigest()


def _classification(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("classification")
    return value if isinstance(value, dict) else {}


def _repair_target(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("repair_target")
    return value if isinstance(value, dict) else {}


def _constraint_delta(record: dict[str, Any]) -> dict[str, Any]:
    value = _repair_target(record).get("constraint_delta")
    return value if isinstance(value, dict) else {}


def _truth_tokens(record: dict[str, Any]) -> list[Any]:
    diagnostics = _classification(record).get("diagnostics")
    if isinstance(diagnostics, dict):
        tokens = diagnostics.get("truth_tokens")
        if isinstance(tokens, list):
            return tokens
    return []


def _truth_source(record: dict[str, Any]) -> str:
    diagnostics = _classification(record).get("diagnostics")
    if isinstance(diagnostics, dict) and isinstance(diagnostics.get("truth_source"), str):
        return diagnostics["truth_source"]
    return "missing"


def _truth_token_kind(record: dict[str, Any]) -> str:
    tokens = _truth_tokens(record)
    if not tokens:
        return "none"
    kinds: set[str] = set()
    for token in tokens:
        token_s = str(token)
        if token_s.startswith("Q"):
            kinds.add("qid")
        elif token_s.startswith("P"):
            kinds.add("pid")
        elif WIKIDATA_DATE_RE.match(token_s):
            kinds.add("date")
        elif any(ch.isdigit() for ch in token_s):
            try:
                float(token_s)
            except ValueError:
                kinds.add("literal")
            else:
                kinds.add("numeric")
        else:
            kinds.add("literal")
    return next(iter(kinds)) if len(kinds) == 1 else "mixed"


def _popularity_bucket(record: dict[str, Any]) -> str:
    pop = record.get("popularity")
    if not isinstance(pop, dict):
        return "unknown"
    bucket = pop.get("bucket") or pop.get("popularity_bucket")
    if isinstance(bucket, str) and bucket in {"head", "mid", "tail", "unknown"}:
        return bucket
    score = pop.get("score")
    if not isinstance(score, (int, float)):
        return "unknown"
    if score >= 2 / 3:
        return "head"
    if score <= 1 / 3:
        return "tail"
    return "mid"


def _constraint_family(record: dict[str, Any]) -> str:
    classification = _classification(record)
    constraint_types = classification.get("constraint_types")
    if isinstance(constraint_types, list):
        for entry in constraint_types:
            if isinstance(entry, dict) and isinstance(entry.get("qid"), str):
                return entry["qid"]
    vc = record.get("violation_context")
    if isinstance(vc, dict) and isinstance(vc.get("report_violation_type_qids"), list):
        for qid in vc["report_violation_type_qids"]:
            if isinstance(qid, str):
                return qid
    changed = _constraint_delta(record).get("changed_constraint_qids") or _constraint_delta(record).get(
        "changed_constraint_types"
    )
    if isinstance(changed, list):
        for qid in changed:
            if isinstance(qid, str):
                return qid
    return "unknown"


def tbox_revision_key(record: dict[str, Any]) -> tuple[str, bool]:
    rt = _repair_target(record)
    delta = _constraint_delta(record)
    property_id = record.get("property") if isinstance(record.get("property"), str) else "UNKNOWN_PROPERTY"
    for key in ("property_revision_id", "property_revision_new", "revision_id"):
        value = rt.get(key)
        if value not in (None, ""):
            return f"TBOX::{property_id}::{value}", False
    value = delta.get("revision_id")
    if value not in (None, ""):
        return f"TBOX::{property_id}::{value}", False
    return f"TBOX::{property_id}::{record.get('id')}", True


def group_key_for_record(record: dict[str, Any]) -> tuple[str, str | None, bool]:
    track = record.get("track")
    rt = _repair_target(record)
    if track == "T_BOX" or rt.get("kind") == "T_BOX":
        key, weak = tbox_revision_key(record)
        return key, key, weak
    qid = record.get("qid")
    property_id = record.get("property")
    if isinstance(qid, str) and qid and isinstance(property_id, str) and property_id:
        return f"ABOX::{qid}::{property_id}", None, False
    return f"ABOX::{record.get('id')}", None, True


def _core_stratum(cls: str, subtype: str) -> str | None:
    if cls == "T_BOX":
        return f"TBOX_{subtype}"
    if cls in {"TypeA", "TypeB", "TypeC"}:
        return f"{cls}_{subtype}"
    return None


def _dev_stratum(cls: str, subtype: str) -> str | None:
    if cls == "TypeA" and subtype in {
        "REJECTION_FORMAT_INVALID",
        "REJECTION_RULE_INVALID",
        "LOGICAL",
        "FORMAT_NORMALIZATION",
        "FORMAT_VALUE_PRUNING",
        "SELF_LINK_REJECTION",
        "SET_MEMBERSHIP_REJECTION",
        "MULTIPLICITY_NORMALIZATION",
        "TARGET_REQUIRED_CLAIM",
    }:
        return "DEV_TYPEA_CLEAN_RULE_REJECTION"
    if cls == "TypeA" and subtype == "DELETE_AMBIGUOUS":
        return "DEV_TYPEA_AMBIGUOUS_DELETE"
    if cls == "TypeB" and subtype.startswith("LOCAL_"):
        return "DEV_TYPEB_LOCAL"
    if cls == "TypeC" and (subtype == "EXTERNAL_BY_ELIMINATION" or subtype.startswith("UNKNOWN_")):
        return "DEV_TYPEC_EXTERNAL_OR_UNKNOWN"
    if cls == "T_BOX" and subtype == "RELAXATION_SET_EXPANSION":
        return "DEV_TBOX_RELAXATION_SET_EXPANSION"
    if cls == "T_BOX" and subtype == "RESTRICTION_SET_CONTRACTION":
        return "DEV_TBOX_RESTRICTION_SET_CONTRACTION"
    if cls == "T_BOX" and subtype == "SCHEMA_UPDATE":
        return "DEV_TBOX_SCHEMA_UPDATE"
    if cls == "T_BOX" and subtype == "COINCIDENTAL_SCHEMA_CHANGE":
        return "DEV_TBOX_COINCIDENTAL_SCHEMA_CHANGE"
    return None


def _analysis_slice(cls: str, subtype: str, stratum: str) -> str:
    if cls == "TypeA" and subtype == "DELETE_AMBIGUOUS":
        return "diagnostic_ambiguous_delete"
    if cls == "TypeA":
        return f"main_ic_l_{subtype.lower()}"
    if cls == "TypeB":
        return f"main_ic_g_{subtype.lower()}"
    if cls == "TypeC" and subtype == "EXTERNAL_BY_ELIMINATION":
        return "main_ic_e_elim"
    if cls == "TypeC" and subtype.startswith("UNKNOWN_"):
        return "diagnostic_ic_u"
    if cls == "T_BOX" and subtype == "COINCIDENTAL_SCHEMA_CHANGE":
        return "diagnostic_tbox_coincidental"
    if cls == "T_BOX" and subtype == "UNKNOWN_TBOX_CAUSALITY":
        return "diagnostic_tbox_unknown_causality"
    if cls == "T_BOX":
        return f"main_tbox_{subtype.lower()}"
    return f"diagnostic_{stratum.lower()}"


def derive_case_metadata(record: dict[str, Any], *, tier: str = "core") -> dict[str, Any] | None:
    case_id = record.get("id")
    if not isinstance(case_id, str) or not case_id:
        return None
    classification = _classification(record)
    cls = classification.get("class") if isinstance(classification.get("class"), str) else "UNKNOWN"
    subtype = classification.get("subtype") if isinstance(classification.get("subtype"), str) else "UNKNOWN"
    confidence = classification.get("confidence") if isinstance(classification.get("confidence"), str) else "unknown"
    track = record.get("track") if isinstance(record.get("track"), str) else "UNKNOWN"
    core_policy_stratum = _core_stratum(cls, subtype)
    selection_stratum = _dev_stratum(cls, subtype) if tier == "dev" else core_policy_stratum
    if selection_stratum is None:
        return None
    group_key, tbox_key, weak_group_key = group_key_for_record(record)
    main_score = (
        core_policy_stratum in MAIN_SCORE_STRATA and confidence != "low" and not subtype.startswith("UNKNOWN_")
    )
    diagnostic_only = not main_score
    return {
        "case_id": case_id,
        "tier": tier,
        "track": track,
        "class": cls,
        "subtype": subtype,
        "confidence": confidence,
        "selection_stratum": selection_stratum,
        "analysis_slice": _analysis_slice(cls, subtype, selection_stratum),
        "analysis_slice_precise": classification.get("analysis_slice_precise") or "",
        "main_score": main_score,
        "diagnostic_only": diagnostic_only,
        "group_key": group_key,
        "tbox_revision_key": tbox_key,
        "weak_group_key": weak_group_key,
        "popularity_bucket": _popularity_bucket(record),
        "constraint_family": _constraint_family(record),
        "decision_constraint_type_qid": classification.get("decision_constraint_type_qid") or "",
        "decision_constraint_type_label": classification.get("decision_constraint_type_label") or "",
        "decision_constraint_source": classification.get("decision_constraint_source") or "",
        "classification_rule_family": classification.get("classification_rule_family") or "",
        "classification_rule_subfamily": classification.get("classification_rule_subfamily") or "",
        "truth_source": _truth_source(record),
        "truth_token_kind": _truth_token_kind(record),
    }


def _load_exclusion_manifest(path: Path | None) -> dict[str, set[str]]:
    if path is None:
        return {"case_ids": set(), "tbox_revision_keys": set(), "abox_group_keys": set()}
    manifest = load_selection_manifest(path)
    annotations = manifest.get("case_annotations") if isinstance(manifest.get("case_annotations"), dict) else {}
    return {
        "case_ids": set(manifest.get("selected_case_ids", [])),
        "tbox_revision_keys": {
            ann.get("tbox_revision_key")
            for ann in annotations.values()
            if isinstance(ann, dict) and isinstance(ann.get("tbox_revision_key"), str)
        },
        "abox_group_keys": {
            ann.get("group_key")
            for ann in annotations.values()
            if isinstance(ann, dict) and isinstance(ann.get("group_key"), str) and ann.get("track") != "T_BOX"
        },
    }


def _round_robin_by_popularity(candidates: list[dict[str, Any]], *, seed: int, tier: str) -> list[dict[str, Any]]:
    buckets: dict[str, deque[dict[str, Any]]] = {}
    for bucket in ("head", "mid", "tail", "unknown"):
        rows = [c for c in candidates if c["popularity_bucket"] == bucket]
        rows.sort(
            key=lambda c: (
                _stable_digest(seed, tier, c["selection_stratum"], c["group_key"], c["case_id"]),
                c["case_id"],
            )
        )
        buckets[bucket] = deque(rows)
    ordered: list[dict[str, Any]] = []
    while any(buckets.values()):
        for bucket in ("head", "mid", "tail", "unknown"):
            if buckets[bucket]:
                ordered.append(buckets[bucket].popleft())
    return ordered


def _select_from_stratum(
    candidates: list[dict[str, Any]],
    *,
    quota: int,
    seed: int,
    tier: str,
    tbox_cap: int,
    abox_cap: int,
    selected_ids: set[str],
    group_counts: Counter[str],
    excluded_case_ids: set[str],
    excluded_tbox_keys: set[str],
    excluded_abox_keys: set[str],
    enforce_abox_exclusion: bool,
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for candidate in _round_robin_by_popularity(candidates, seed=seed, tier=tier):
        if len(chosen) >= quota:
            break
        case_id = candidate["case_id"]
        group_key = candidate["group_key"]
        tbox_key = candidate.get("tbox_revision_key")
        if case_id in selected_ids or case_id in excluded_case_ids:
            continue
        if isinstance(tbox_key, str) and tbox_key in excluded_tbox_keys:
            continue
        if enforce_abox_exclusion and candidate["track"] != "T_BOX" and group_key in excluded_abox_keys:
            continue
        cap = tbox_cap if candidate["track"] == "T_BOX" else abox_cap
        if group_counts[group_key] >= cap:
            continue
        selected_ids.add(case_id)
        group_counts[group_key] += 1
        chosen.append(candidate)
    return chosen


def _count_by(annotations: dict[str, dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(ann.get(field)) for ann in annotations.values()).items()))


def _validate_manifest(
    *,
    selected: list[dict[str, Any]],
    main_ids: list[str],
    diagnostic_ids: list[str],
    tbox_cap: int,
    abox_cap: int,
    exclude: dict[str, set[str]],
) -> dict[str, Any]:
    selected_ids = [c["case_id"] for c in selected]
    annotations = {c["case_id"]: c for c in selected}
    tbox_counts = Counter(c["tbox_revision_key"] for c in selected if isinstance(c.get("tbox_revision_key"), str))
    abox_counts = Counter(c["group_key"] for c in selected if c.get("track") != "T_BOX")
    main_set = set(main_ids)
    diagnostic_set = set(diagnostic_ids)
    selected_set = set(selected_ids)
    dev_case_overlap = len(selected_set & exclude["case_ids"])
    dev_tbox_overlap = len({k for k in tbox_counts if k in exclude["tbox_revision_keys"]})
    dev_abox_overlap = len({k for k in abox_counts if k in exclude["abox_group_keys"]})
    unknown_or_low = sum(
        1
        for cid in main_ids
        if annotations[cid]["confidence"] == "low" or str(annotations[cid]["subtype"]).startswith("UNKNOWN_")
    )
    diagnostic_in_main = sum(
        1
        for cid in main_ids
        if annotations[cid]["subtype"] in DIAGNOSTIC_SUBTYPES or not annotations[cid]["main_score"]
    )
    max_tbox = max(tbox_counts.values(), default=0)
    max_abox = max(abox_counts.values(), default=0)
    return {
        "selected_case_count_matches": len(selected_ids) == len(selected_set),
        "selected_case_ids_unique": len(selected_ids) == len(selected_set),
        "main_plus_diagnostic_equals_selected": main_set.isdisjoint(diagnostic_set)
        and main_set | diagnostic_set == selected_set,
        "max_tbox_per_revision": max_tbox,
        "max_abox_per_qid_property": max_abox,
        "dev_core_case_overlap": dev_case_overlap,
        "dev_core_tbox_revision_overlap": dev_tbox_overlap,
        "dev_core_abox_group_overlap": dev_abox_overlap,
        "unknown_or_low_confidence_in_main_score": unknown_or_low,
        "diagnostic_subtypes_in_main_score": diagnostic_in_main,
        "hard_validation_passed": (
            dev_case_overlap == 0
            and dev_tbox_overlap == 0
            and max_tbox <= tbox_cap
            and unknown_or_low == 0
            and diagnostic_in_main == 0
        ),
    }


def build_tier_manifest(options: SelectionOptions) -> dict[str, Any]:
    if options.tier not in {"core", "dev"}:
        raise ValueError("tier must be 'core' or 'dev'.")
    if options.selected_case_order not in SUPPORTED_SELECTED_CASE_ORDERS:
        raise ValueError(f"selected_case_order must be one of {sorted(SUPPORTED_SELECTED_CASE_ORDERS)!r}.")

    target_size = options.dev_size if options.tier == "dev" else options.core_size
    quotas = dict(options.quotas or (DEV_QUOTAS if options.tier == "dev" else CORE_QUOTAS))
    tbox_cap = options.tbox_cap_dev if options.tier == "dev" else options.tbox_cap_core
    abox_cap = options.abox_cap_dev if options.tier == "dev" else options.abox_cap_core
    exclude = _load_exclusion_manifest(options.exclude_manifest)

    candidates_by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_records = 0
    with options.classified_benchmark.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if options.progress_every > 0 and line_number % options.progress_every == 0:
                print(f"[progress] scanned={line_number:,} selected_tier={options.tier}")
            if not line.strip():
                continue
            record = json.loads(line)
            total_records += 1
            metadata = derive_case_metadata(record, tier=options.tier)
            if metadata is not None:
                candidates_by_stratum[metadata["selection_stratum"]].append(metadata)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    group_counts: Counter[str] = Counter()
    underfilled: list[dict[str, Any]] = []
    warnings: list[str] = []

    for stratum, quota in quotas.items():
        chosen = _select_from_stratum(
            candidates_by_stratum.get(stratum, []),
            quota=quota,
            seed=options.seed,
            tier=options.tier,
            tbox_cap=tbox_cap,
            abox_cap=abox_cap,
            selected_ids=selected_ids,
            group_counts=group_counts,
            excluded_case_ids=exclude["case_ids"],
            excluded_tbox_keys=exclude["tbox_revision_keys"],
            excluded_abox_keys=exclude["abox_group_keys"],
            enforce_abox_exclusion=options.tier == "core",
        )
        selected.extend(chosen)
        remaining = quota - len(chosen)
        if remaining <= 0:
            continue
        underfilled.append({"selection_stratum": stratum, "quota": quota, "selected": len(chosen)})
        for backfill_stratum in CORE_BACKFILL.get(stratum, []):
            if remaining <= 0:
                break
            backfill = _select_from_stratum(
                candidates_by_stratum.get(backfill_stratum, []),
                quota=remaining,
                seed=options.seed,
                tier=options.tier,
                tbox_cap=tbox_cap,
                abox_cap=abox_cap,
                selected_ids=selected_ids,
                group_counts=group_counts,
                excluded_case_ids=exclude["case_ids"],
                excluded_tbox_keys=exclude["tbox_revision_keys"],
                excluded_abox_keys=exclude["abox_group_keys"],
                enforce_abox_exclusion=options.tier == "core",
            )
            for item in backfill:
                item = dict(item)
                item["quota_backfill_for"] = stratum
                selected.append(item)
            remaining -= len(backfill)

    if len(selected) < target_size:
        selected_strata = set(quotas)
        pool = [c for rows in candidates_by_stratum.values() for c in rows if c["selection_stratum"] in selected_strata]
        backfill = _select_from_stratum(
            pool,
            quota=target_size - len(selected),
            seed=options.seed,
            tier=options.tier,
            tbox_cap=tbox_cap,
            abox_cap=abox_cap,
            selected_ids=selected_ids,
            group_counts=group_counts,
            excluded_case_ids=exclude["case_ids"],
            excluded_tbox_keys=exclude["tbox_revision_keys"],
            excluded_abox_keys=exclude["abox_group_keys"],
            enforce_abox_exclusion=options.tier == "core",
        )
        selected.extend(backfill)

    if options.tier == "core" and len(selected) < target_size:
        warnings.append("Core quotas could not be satisfied with dev A-box group exclusion; retrying with A-box overlap allowed.")
        pool = [c for rows in candidates_by_stratum.values() for c in rows if c["case_id"] not in selected_ids]
        backfill = _select_from_stratum(
            pool,
            quota=target_size - len(selected),
            seed=options.seed,
            tier=options.tier,
            tbox_cap=tbox_cap,
            abox_cap=abox_cap,
            selected_ids=selected_ids,
            group_counts=group_counts,
            excluded_case_ids=exclude["case_ids"],
            excluded_tbox_keys=exclude["tbox_revision_keys"],
            excluded_abox_keys=exclude["abox_group_keys"],
            enforce_abox_exclusion=False,
        )
        selected.extend(backfill)

    annotations = {c["case_id"]: c for c in selected}
    selected_case_ids = sorted(
        annotations,
        key=lambda cid: _selected_case_sort_key(
            cid,
            annotations,
            seed=options.seed,
            tier=options.tier,
            selected_case_order=options.selected_case_order,
        ),
    )
    main_score_case_ids = [cid for cid in selected_case_ids if annotations[cid]["main_score"]]
    diagnostic_case_ids = [cid for cid in selected_case_ids if annotations[cid]["diagnostic_only"]]
    validation = _validate_manifest(
        selected=selected,
        main_ids=main_score_case_ids,
        diagnostic_ids=diagnostic_case_ids,
        tbox_cap=tbox_cap,
        abox_cap=abox_cap,
        exclude=exclude,
    )
    if validation["dev_core_abox_group_overlap"]:
        warnings.append(f"dev_core_abox_group_overlap={validation['dev_core_abox_group_overlap']}")

    return {
        "manifest_type": "benchmark_selection",
        "manifest_version": "phase_c_v1",
        "tier": options.tier,
        "seed": options.seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "inputs": {
            "classified_benchmark": str(options.classified_benchmark),
            "exclude_manifest": str(options.exclude_manifest) if options.exclude_manifest else None,
        },
        "policy": {
            "target_size": target_size,
            "quotas": quotas,
            "tbox_cap_per_property_revision": tbox_cap,
            "abox_cap_per_qid_property": abox_cap,
            "selected_case_order": options.selected_case_order,
            "stable_ordering": "sha1(seed|tier|selection_stratum|group_key|case_id)",
            "typec_semantics": "EXTERNAL_BY_ELIMINATION is IC-E-elim/no-retrieval stress, not confirmed external evidence.",
        },
        "selected_case_ids": selected_case_ids,
        "main_score_case_ids": main_score_case_ids,
        "diagnostic_case_ids": diagnostic_case_ids,
        "case_annotations": {cid: annotations[cid] for cid in selected_case_ids},
        "counts": {
            "input_records_scanned": total_records,
            "selected": len(selected_case_ids),
            "main_score": len(main_score_case_ids),
            "diagnostic": len(diagnostic_case_ids),
            "by_track": _count_by(annotations, "track"),
            "by_class": _count_by(annotations, "class"),
            "by_subtype": _count_by(annotations, "subtype"),
            "by_selection_stratum": _count_by(annotations, "selection_stratum"),
            "by_confidence": _count_by(annotations, "confidence"),
            "by_popularity_bucket": _count_by(annotations, "popularity_bucket"),
            "by_constraint_family": _count_by(annotations, "constraint_family"),
        },
        "underfilled_quotas": underfilled,
        "warnings": warnings,
        "validation": validation,
    }


def build_selection_manifest(
    classified_path: str | Path,
    *,
    tbox_cap_per_update: int = 100,
    seed: int = DEFAULT_SELECTION_SEED,
    selected_case_order: str = DEFAULT_SELECTED_CASE_ORDER,
    progress_every: int = 0,
) -> dict[str, Any]:
    if selected_case_order not in SUPPORTED_SELECTED_CASE_ORDERS:
        raise ValueError(f"selected_case_order must be one of {sorted(SUPPORTED_SELECTED_CASE_ORDERS)!r}.")

    selected_a_box: list[str] = []
    tbox_by_revision: dict[str, list[tuple[str, str]]] = defaultdict(list)
    total_records = 0

    with Path(classified_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if progress_every > 0 and line_number % progress_every == 0:
                print(f"[progress] scanned={line_number:,}")
            if not line.strip():
                continue
            total_records += 1
            case_match = CASE_ID_RE.search(line)
            if case_match is None:
                continue
            case_id = case_match.group(1)
            if TRACK_TBOX_MARKER not in line:
                selected_a_box.append(case_id)
                continue
            revision_match = PROPERTY_REVISION_RE.search(line)
            revision_id = revision_match.group(1) if revision_match else case_id
            tbox_by_revision[revision_id].append((_legacy_tbox_rank(case_id, revision_id, seed), case_id))

    selected_tbox: list[str] = []
    tbox_selected_counts: dict[str, int] = {}
    for revision_id, ranked_cases in sorted(tbox_by_revision.items()):
        ranked_cases.sort()
        chosen = [case_id for _, case_id in ranked_cases[:tbox_cap_per_update]]
        selected_tbox.extend(chosen)
        tbox_selected_counts[revision_id] = len(chosen)

    selected_case_ids = sorted(
        selected_a_box + selected_tbox,
        key=lambda case_id: _legacy_selected_case_sort_key(case_id, seed, selected_case_order),
    )

    return {
        "manifest_type": "benchmark_case_selection",
        "manifest_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "inputs": {"classified_benchmark": str(classified_path)},
        "policy": {
            "tbox_cap_per_update": tbox_cap_per_update,
            "seed": seed,
            "selected_case_order": selected_case_order,
            "selected_case_ids_ordering": (
                "case_id ascending"
                if selected_case_order == "sorted"
                else "sha1(seed|selected_case_ids|case_id) ascending"
            ),
        },
        "selected_case_ids": selected_case_ids,
        "counts": {
            "input_records_scanned": total_records,
            "selected_cases": len(selected_case_ids),
            "selected_a_box_cases": len(selected_a_box),
            "selected_t_box_cases": len(selected_tbox),
            "distinct_t_box_updates": len(tbox_by_revision),
            "t_box_selected_counts_by_update": tbox_selected_counts,
        },
    }


def load_selection_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Selection manifest must be a JSON object.")
    selected_case_ids = manifest.get("selected_case_ids")
    if not isinstance(selected_case_ids, list):
        raise ValueError("Selection manifest must contain selected_case_ids.")
    normalized_ids = []
    seen = set()
    for case_id in selected_case_ids:
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Selection manifest contains an invalid case id.")
        if case_id in seen:
            raise ValueError(f"Selection manifest contains duplicate case id: {case_id}")
        seen.add(case_id)
        normalized_ids.append(case_id)
    manifest["selected_case_ids"] = normalized_ids
    return manifest


def resolve_case_id_filter(
    *,
    case_ids: Optional[Iterable[str]] = None,
    selection_manifest_path: str | Path | None = None,
) -> Optional[list[str]]:
    explicit_ids = _ordered_unique_case_ids(case_ids)
    explicit_id_set = set(explicit_ids)
    manifest_ids: list[str] | None = None
    if selection_manifest_path:
        manifest = load_selection_manifest(selection_manifest_path)
        manifest_ids = _ordered_unique_case_ids(manifest.get("selected_case_ids"))

    if not explicit_ids and manifest_ids is None:
        return None
    if not explicit_ids:
        return list(manifest_ids or [])
    if manifest_ids is None:
        return explicit_ids
    return [case_id for case_id in manifest_ids if case_id in explicit_id_set]
