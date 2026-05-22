from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from lib.benchmark_selection import derive_case_metadata, load_selection_manifest

AUDIT_POLICY_VERSION = "phase_d_v1"
DEFAULT_AUDIT_SEED = 13
DEFAULT_AUDIT_SIZE = 450
DEFAULT_TBOX_CAP_PER_REVISION = 5
DEFAULT_ABOX_CAP_PER_GROUP = 3

AUDIT_QUOTAS: dict[str, int] = {
    "TypeC_EXTERNAL_BY_ELIMINATION_QID_TRUTH": 50,
    "TypeC_EXTERNAL_BY_ELIMINATION_LITERAL_TRUTH": 50,
    "TypeC_UNKNOWN_OR_SPARSE_DIAGNOSTIC": 30,
    "TypeA_REJECTION_FORMAT_INVALID": 50,
    "TypeA_DELETE_AMBIGUOUS": 50,
    "TypeB_LOCAL_TEXT": 50,
    "TypeB_LOCAL_FOCUS_PREREPAIR_PROPERTY": 40,
    "TBOX_SCHEMA_UPDATE": 50,
    "TBOX_COINCIDENTAL_SCHEMA_CHANGE": 40,
    "TBOX_DIRECTIONAL_RELAXATION_OR_RESTRICTION": 40,
}

AUDIT_FIELDNAMES = [
    "case_id",
    "qid",
    "property",
    "track",
    "class",
    "subtype",
    "confidence",
    "selection_stratum",
    "analysis_slice",
    "main_score",
    "diagnostic_only",
    "popularity_bucket",
    "constraint_family",
    "truth_source",
    "truth_token_kind",
    "truth_tokens_preview",
    "decision_branch",
    "local_match_kind",
    "local_match_source",
    "tbox_revision_key",
    "group_key",
    "repair_locus_correct",
    "historical_target_well_defined",
    "target_visible_locally",
    "extractor_missed_local_evidence",
    "external_evidence_required",
    "typec_judgment",
    "typea_judgment",
    "typeb_judgment",
    "tbox_judgment",
    "core_recommendation",
    "notes",
    "annotator_id",
    "annotation_timestamp_utc",
]

HUMAN_ALLOWED_VALUES: dict[str, list[str]] = {
    "repair_locus_correct": ["yes", "no", "unclear"],
    "historical_target_well_defined": ["yes", "no", "unclear"],
    "target_visible_locally": ["yes", "no", "partial", "unclear"],
    "extractor_missed_local_evidence": ["yes", "no", "unclear", "not_applicable"],
    "external_evidence_required": ["yes", "no", "maybe", "unclear", "not_applicable"],
    "typec_judgment": [
        "external_confirmed",
        "external_by_elimination_ok",
        "local_missed",
        "unknown_or_incomplete",
        "bad_target",
        "not_typec",
    ],
    "typea_judgment": [
        "clean_rule_or_format",
        "delete_ambiguous_ok",
        "needs_local_evidence",
        "needs_external_evidence",
        "overclaimed",
        "not_typea",
    ],
    "typeb_judgment": [
        "local_confirmed",
        "local_false_positive",
        "leakage_suspected",
        "weak_literal_match",
        "not_typeb",
    ],
    "tbox_judgment": [
        "causal_schema_repair",
        "plausible_schema_update",
        "coincidental_or_weak",
        "wrong_constraint_family",
        "not_tbox",
    ],
    "core_recommendation": ["main", "diagnostic", "exclude", "needs_discussion"],
}

UNANNOTATED = "unannotated"
ANNOTATION_FIELDS = list(HUMAN_ALLOWED_VALUES) + ["notes", "annotator_id", "annotation_timestamp_utc"]
TRUTH_TOKEN_KINDS = ["qid", "literal", "date", "numeric", "mixed", "none_expected"]
POPULARITY_BUCKETS = ["head", "mid", "tail", "unknown"]


@dataclass(frozen=True)
class AuditBuildOptions:
    classified_benchmark: Path
    core_manifest: Path
    dev_manifest: Path | None
    seed: int = DEFAULT_AUDIT_SEED
    quotas: dict[str, int] | None = None
    tbox_cap_per_revision: int = DEFAULT_TBOX_CAP_PER_REVISION
    abox_cap_per_group: int = DEFAULT_ABOX_CAP_PER_GROUP
    progress_every: int = 100000


def _classification(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("classification")
    return value if isinstance(value, dict) else {}


def _diagnostics(record: dict[str, Any]) -> dict[str, Any]:
    value = _classification(record).get("diagnostics")
    return value if isinstance(value, dict) else {}


def _decision_trace(record: dict[str, Any]) -> list[dict[str, Any]]:
    value = _classification(record).get("decision_trace")
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _trace_step(record: dict[str, Any], step_name: str) -> dict[str, Any] | None:
    for step in _decision_trace(record):
        if step.get("step") == step_name:
            return step
    return None


def decision_branch(record: dict[str, Any]) -> str:
    step = _trace_step(record, "branch")
    value = step.get("result") if isinstance(step, dict) else None
    return value if isinstance(value, str) else ""


def local_match_details(record: dict[str, Any]) -> tuple[str, str]:
    step = _trace_step(record, "local_availability")
    evidence = step.get("evidence") if isinstance(step, dict) else None
    if not isinstance(evidence, dict):
        return "", ""
    matches = evidence.get("matches")
    if isinstance(matches, list) and matches:
        first = next((m for m in matches if isinstance(m, dict)), None)
        if isinstance(first, dict):
            return str(first.get("kind") or ""), str(first.get("source") or "")
    sources = evidence.get("sources_used")
    if isinstance(sources, list) and sources:
        return "", str(sources[0])
    return "", ""


def local_context_score(record: dict[str, Any]) -> int:
    step = _trace_step(record, "local_availability")
    evidence = step.get("evidence") if isinstance(step, dict) else None
    if isinstance(evidence, dict) and isinstance(evidence.get("local_ids_count"), int):
        return evidence["local_ids_count"]
    return 10**9


def truth_tokens(record: dict[str, Any]) -> list[Any]:
    tokens = _diagnostics(record).get("truth_tokens")
    return tokens if isinstance(tokens, list) else []


def audit_truth_token_kind(record: dict[str, Any]) -> str:
    tokens = truth_tokens(record)
    if not tokens:
        return "none_expected"
    kinds: set[str] = set()
    for token in tokens:
        token_s = str(token)
        if token_s.startswith("Q"):
            kinds.add("qid")
        elif token_s.startswith(("+", "-")) and "-" in token_s:
            kinds.add("date")
        else:
            try:
                float(token_s)
            except ValueError:
                kinds.add("literal")
            else:
                kinds.add("numeric")
    return next(iter(kinds)) if len(kinds) == 1 else "mixed"


def truth_tokens_preview(record: dict[str, Any], *, limit: int = 5) -> str:
    tokens = [str(token) for token in truth_tokens(record)]
    if len(tokens) <= limit:
        return json.dumps(tokens, ensure_ascii=False)
    preview = tokens[:limit] + [f"...(+{len(tokens) - limit})"]
    return json.dumps(preview, ensure_ascii=False)


def _stable_digest(seed: int, audit_stratum: str, group_key: str, case_id: str) -> str:
    return hashlib.sha1(f"{seed}|audit|{audit_stratum}|{group_key}|{case_id}".encode()).hexdigest()


def _round_robin_by_popularity(candidates: list[dict[str, Any]], *, seed: int, stratum: str) -> list[dict[str, Any]]:
    buckets: dict[str, deque[dict[str, Any]]] = {}
    for bucket in POPULARITY_BUCKETS:
        rows = [row for row in candidates if row.get("popularity_bucket") == bucket]
        rows.sort(key=lambda row: (_stable_digest(seed, stratum, row["group_key"], row["case_id"]), row["case_id"]))
        buckets[bucket] = deque(rows)
    ordered: list[dict[str, Any]] = []
    while any(buckets.values()):
        for bucket in POPULARITY_BUCKETS:
            if buckets[bucket]:
                ordered.append(buckets[bucket].popleft())
    return ordered


def audit_stratum_for_record(record: dict[str, Any]) -> str | None:
    classification = _classification(record)
    cls = classification.get("class")
    subtype = classification.get("subtype")
    truth_kind = audit_truth_token_kind(record)
    if cls == "TypeC" and subtype == "EXTERNAL_BY_ELIMINATION" and truth_kind == "qid":
        return "TypeC_EXTERNAL_BY_ELIMINATION_QID_TRUTH"
    if cls == "TypeC" and subtype == "EXTERNAL_BY_ELIMINATION" and truth_kind != "qid":
        return "TypeC_EXTERNAL_BY_ELIMINATION_LITERAL_TRUTH"
    if cls == "TypeC" and isinstance(subtype, str) and subtype.startswith("UNKNOWN_"):
        return "TypeC_UNKNOWN_OR_SPARSE_DIAGNOSTIC"
    if cls == "TypeA" and subtype == "REJECTION_FORMAT_INVALID":
        return "TypeA_REJECTION_FORMAT_INVALID"
    if cls == "TypeA" and subtype == "DELETE_AMBIGUOUS":
        return "TypeA_DELETE_AMBIGUOUS"
    if cls == "TypeB" and subtype == "LOCAL_TEXT":
        return "TypeB_LOCAL_TEXT"
    if cls == "TypeB" and subtype == "LOCAL_FOCUS_PREREPAIR_PROPERTY":
        return "TypeB_LOCAL_FOCUS_PREREPAIR_PROPERTY"
    if cls == "T_BOX" and subtype == "SCHEMA_UPDATE":
        return "TBOX_SCHEMA_UPDATE"
    if cls == "T_BOX" and subtype == "COINCIDENTAL_SCHEMA_CHANGE":
        return "TBOX_COINCIDENTAL_SCHEMA_CHANGE"
    if cls == "T_BOX" and subtype in {"RELAXATION_SET_EXPANSION", "RESTRICTION_SET_CONTRACTION"}:
        return "TBOX_DIRECTIONAL_RELAXATION_OR_RESTRICTION"
    return None


def _annotation_defaults() -> dict[str, str]:
    values = {field: UNANNOTATED for field in HUMAN_ALLOWED_VALUES}
    values.update({"notes": "", "annotator_id": "", "annotation_timestamp_utc": ""})
    return values


def make_audit_row(
    record: dict[str, Any],
    *,
    audit_stratum: str,
    manifest_annotation: dict[str, Any] | None,
    tier_hint: str,
) -> dict[str, Any] | None:
    metadata = dict(manifest_annotation or (derive_case_metadata(record, tier="core") or {}))
    if not metadata:
        return None
    diagnostics = _diagnostics(record)
    local_kind, local_source = local_match_details(record)
    row: dict[str, Any] = {
        "case_id": metadata["case_id"],
        "qid": record.get("qid") or "",
        "property": record.get("property") or "",
        "track": metadata.get("track") or record.get("track") or "",
        "class": metadata.get("class") or _classification(record).get("class") or "",
        "subtype": metadata.get("subtype") or _classification(record).get("subtype") or "",
        "confidence": metadata.get("confidence") or _classification(record).get("confidence") or "",
        "selection_stratum": audit_stratum,
        "analysis_slice": metadata.get("analysis_slice") or "",
        "main_score": bool(metadata.get("main_score")),
        "diagnostic_only": bool(metadata.get("diagnostic_only")),
        "popularity_bucket": metadata.get("popularity_bucket") or "unknown",
        "constraint_family": metadata.get("constraint_family") or "",
        "truth_source": diagnostics.get("truth_source") or metadata.get("truth_source") or "",
        "truth_token_kind": audit_truth_token_kind(record),
        "truth_tokens_preview": truth_tokens_preview(record),
        "decision_branch": decision_branch(record),
        "local_match_kind": local_kind,
        "local_match_source": local_source,
        "tbox_revision_key": metadata.get("tbox_revision_key") or "",
        "group_key": metadata.get("group_key") or "",
        "_tier_hint": tier_hint,
        "_local_context_score": local_context_score(record),
    }
    row.update(_annotation_defaults())
    return row


def _manifest_annotations(path: Path | None) -> tuple[set[str], dict[str, dict[str, Any]]]:
    if path is None:
        return set(), {}
    manifest = load_selection_manifest(path)
    annotations = manifest.get("case_annotations") if isinstance(manifest.get("case_annotations"), dict) else {}
    normalized = {case_id: ann for case_id, ann in annotations.items() if isinstance(ann, dict)}
    return set(manifest.get("selected_case_ids", [])), normalized


def _load_candidates(options: AuditBuildOptions) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    core_ids, core_annotations = _manifest_annotations(options.core_manifest)
    dev_ids, _ = _manifest_annotations(options.dev_manifest)
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scanned = 0
    matched = 0
    with options.classified_benchmark.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if options.progress_every > 0 and line_number % options.progress_every == 0:
                print(f"[progress] scanned={line_number:,}")
            if not line.strip():
                continue
            scanned += 1
            record = json.loads(line)
            case_id = record.get("id")
            if not isinstance(case_id, str) or not case_id:
                continue
            stratum = audit_stratum_for_record(record)
            if stratum is None:
                continue
            tier_hint = "core" if case_id in core_ids else "dev" if case_id in dev_ids else "full"
            row = make_audit_row(
                record,
                audit_stratum=stratum,
                manifest_annotation=core_annotations.get(case_id),
                tier_hint=tier_hint,
            )
            if row is None:
                continue
            row["_dev_overlap"] = case_id in dev_ids
            candidates[stratum].append(row)
            matched += 1
    return candidates, {"input_records_scanned": scanned, "candidate_records": matched}


def _candidate_sort_key(row: dict[str, Any], *, seed: int, stratum: str) -> tuple[int, str, int, str]:
    tier_rank = {"core": 0, "full": 1, "dev": 2}.get(str(row.get("_tier_hint")), 3)
    sparse_rank = int(row.get("_local_context_score") or 10**9)
    return (
        tier_rank,
        _stable_digest(seed, stratum, row["group_key"], row["case_id"]),
        sparse_rank,
        row["case_id"],
    )


def _can_select(
    row: dict[str, Any],
    *,
    selected_ids: set[str],
    group_counts: Counter[str],
    tbox_cap: int,
    abox_cap: int,
    allow_dev_overlap: bool,
) -> bool:
    if row["case_id"] in selected_ids:
        return False
    if row.get("_dev_overlap") and not allow_dev_overlap:
        return False
    if row["track"] == "T_BOX":
        key = row["tbox_revision_key"] or row["group_key"]
        return group_counts[key] < tbox_cap
    return group_counts[row["group_key"]] < abox_cap


def _mark_selected(row: dict[str, Any], *, selected_ids: set[str], group_counts: Counter[str]) -> None:
    selected_ids.add(row["case_id"])
    key = row["tbox_revision_key"] or row["group_key"] if row["track"] == "T_BOX" else row["group_key"]
    group_counts[key] += 1


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    quota: int,
    seed: int,
    stratum: str,
    selected_ids: set[str],
    group_counts: Counter[str],
    tbox_cap: int,
    abox_cap: int,
    allow_dev_overlap: bool,
    sparse_first: bool = False,
) -> list[dict[str, Any]]:
    if sparse_first:
        ordered = sorted(
            rows,
            key=lambda row: (
                int(row.get("_local_context_score") or 10**9),
                _candidate_sort_key(row, seed=seed, stratum=stratum),
            ),
        )
    else:
        ordered = []
        for tier in ("core", "full", "dev"):
            tier_rows = [row for row in rows if row.get("_tier_hint") == tier]
            ordered.extend(_round_robin_by_popularity(tier_rows, seed=seed, stratum=stratum))
    chosen: list[dict[str, Any]] = []
    for row in ordered:
        if len(chosen) >= quota:
            break
        if not _can_select(
            row,
            selected_ids=selected_ids,
            group_counts=group_counts,
            tbox_cap=tbox_cap,
            abox_cap=abox_cap,
            allow_dev_overlap=allow_dev_overlap,
        ):
            continue
        _mark_selected(row, selected_ids=selected_ids, group_counts=group_counts)
        chosen.append(row)
    return chosen


def build_audit_sample(options: AuditBuildOptions) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    quotas = dict(options.quotas or AUDIT_QUOTAS)
    candidates_by_stratum, scan_counts = _load_candidates(options)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    group_counts: Counter[str] = Counter()
    underfilled: list[dict[str, Any]] = []
    warnings: list[str] = []

    for stratum, quota in quotas.items():
        chosen = _select_rows(
            candidates_by_stratum.get(stratum, []),
            quota=quota,
            seed=options.seed,
            stratum=stratum,
            selected_ids=selected_ids,
            group_counts=group_counts,
            tbox_cap=options.tbox_cap_per_revision,
            abox_cap=options.abox_cap_per_group,
            allow_dev_overlap=False,
            sparse_first=stratum == "TypeC_UNKNOWN_OR_SPARSE_DIAGNOSTIC",
        )
        selected.extend(chosen)
        if len(chosen) >= quota:
            continue

        underfilled.append({"selection_stratum": stratum, "quota": quota, "selected": len(chosen)})
        warnings.append(f"{stratum} underfilled: selected {len(chosen)} of requested {quota}.")
        remaining = quota - len(chosen)
        if stratum == "TypeC_UNKNOWN_OR_SPARSE_DIAGNOSTIC":
            warnings.append(
                "TypeC_UNKNOWN_OR_SPARSE_DIAGNOSTIC underfilled; backfilled with weakest "
                "TypeC_EXTERNAL_BY_ELIMINATION local-context diagnostics."
            )
            pool = (
                candidates_by_stratum.get("TypeC_EXTERNAL_BY_ELIMINATION_QID_TRUTH", [])
                + candidates_by_stratum.get("TypeC_EXTERNAL_BY_ELIMINATION_LITERAL_TRUTH", [])
            )
            backfill = _select_rows(
                pool,
                quota=remaining,
                seed=options.seed,
                stratum=stratum,
                selected_ids=selected_ids,
                group_counts=group_counts,
                tbox_cap=options.tbox_cap_per_revision,
                abox_cap=options.abox_cap_per_group,
                allow_dev_overlap=False,
                sparse_first=True,
            )
            for row in backfill:
                row["quota_backfill_for"] = stratum
                row["selection_stratum"] = stratum
            selected.extend(backfill)

    target_size = sum(quotas.values())
    if len(selected) < target_size:
        remaining = target_size - len(selected)
        pool = [row for rows in candidates_by_stratum.values() for row in rows]
        backfill = _select_rows(
            pool,
            quota=remaining,
            seed=options.seed,
            stratum="GLOBAL_BACKFILL",
            selected_ids=selected_ids,
            group_counts=group_counts,
            tbox_cap=options.tbox_cap_per_revision,
            abox_cap=options.abox_cap_per_group,
            allow_dev_overlap=False,
        )
        for row in backfill:
            row.setdefault("quota_backfill_for", "GLOBAL_BACKFILL")
        selected.extend(backfill)

    if len(selected) < target_size:
        warnings.append("Audit sample could not be filled without dev overlap; retrying remaining slots with dev overlap.")
        pool = [row for rows in candidates_by_stratum.values() for row in rows]
        backfill = _select_rows(
            pool,
            quota=target_size - len(selected),
            seed=options.seed,
            stratum="DEV_OVERLAP_BACKFILL",
            selected_ids=selected_ids,
            group_counts=group_counts,
            tbox_cap=options.tbox_cap_per_revision,
            abox_cap=options.abox_cap_per_group,
            allow_dev_overlap=True,
        )
        for row in backfill:
            row.setdefault("quota_backfill_for", "DEV_OVERLAP_BACKFILL")
        selected.extend(backfill)

    selected.sort(key=lambda row: (_stable_digest(options.seed, row["selection_stratum"], row["group_key"], row["case_id"]), row["case_id"]))
    public_rows = [{field: row.get(field, "") for field in AUDIT_FIELDNAMES} for row in selected]
    metadata = {
        "manifest_type": "manual_audit_sample",
        "policy_version": AUDIT_POLICY_VERSION,
        "seed": options.seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "inputs": {
            "classified_benchmark": str(options.classified_benchmark),
            "core_manifest": str(options.core_manifest),
            "dev_manifest": str(options.dev_manifest) if options.dev_manifest else None,
        },
        "quotas": quotas,
        "counts": {
            **scan_counts,
            "selected": len(public_rows),
            "by_selection_stratum": dict(sorted(Counter(row["selection_stratum"] for row in public_rows).items())),
            "dev_overlap": sum(1 for row in selected if row.get("_dev_overlap")),
            "max_tbox_per_revision": max(
                Counter((row["tbox_revision_key"] or row["group_key"]) for row in public_rows if row["track"] == "T_BOX").values(),
                default=0,
            ),
            "max_abox_per_group": max(
                Counter(row["group_key"] for row in public_rows if row["track"] != "T_BOX").values(),
                default=0,
            ),
        },
        "underfilled_quotas": underfilled,
        "warnings": warnings,
    }
    if metadata["counts"]["dev_overlap"]:
        metadata["warnings"].append(f"dev_overlap={metadata['counts']['dev_overlap']}")
    return public_rows, metadata


def audit_annotation_schema(metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for field in AUDIT_FIELDNAMES:
        if field in {"main_score", "diagnostic_only"}:
            properties[field] = {"type": "boolean"}
        elif field in HUMAN_ALLOWED_VALUES:
            properties[field] = {"type": "string", "enum": HUMAN_ALLOWED_VALUES[field] + [UNANNOTATED, ""]}
        elif field == "popularity_bucket":
            properties[field] = {"type": "string", "enum": POPULARITY_BUCKETS}
        elif field == "truth_token_kind":
            properties[field] = {"type": "string", "enum": TRUTH_TOKEN_KINDS}
        else:
            properties[field] = {"type": "string"}
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "WikidataRepairEval Phase D manual audit row",
        "type": "object",
        "additionalProperties": False,
        "required": AUDIT_FIELDNAMES,
        "properties": properties,
        "allowed_values": HUMAN_ALLOWED_VALUES,
    }
    if metadata is not None:
        schema["sampling_metadata"] = metadata
    return schema


def write_audit_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_audit_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_schema(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit_annotation_schema(metadata), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_annotation_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def is_annotated_value(value: Any) -> bool:
    return str(value or "").strip() not in {"", UNANNOTATED}


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _metric_from_rows(rows: list[dict[str, str]], field: str, good_values: set[str]) -> dict[str, Any]:
    annotated = [row for row in rows if is_annotated_value(row.get(field))]
    numerator = sum(1 for row in annotated if row.get(field) in good_values)
    return {"numerator": numerator, "denominator": len(annotated), "rate": _rate(numerator, len(annotated))}


def _bool_from_csv(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def summarize_annotations(rows: list[dict[str, str]]) -> dict[str, Any]:
    total = len(rows)
    complete_rows = [
        row
        for row in rows
        if all(is_annotated_value(row.get(field)) for field in HUMAN_ALLOWED_VALUES)
    ]
    unannotated_rows = [row for row in rows if not any(is_annotated_value(row.get(field)) for field in HUMAN_ALLOWED_VALUES)]

    typec_external = [
        row
        for row in rows
        if row.get("class") == "TypeC" and row.get("subtype") == "EXTERNAL_BY_ELIMINATION"
    ]
    typec_all = [row for row in rows if row.get("class") == "TypeC"]
    typea_format = [row for row in rows if row.get("class") == "TypeA" and row.get("subtype") == "REJECTION_FORMAT_INVALID"]
    typea_delete = [row for row in rows if row.get("class") == "TypeA" and row.get("subtype") == "DELETE_AMBIGUOUS"]
    typeb = [row for row in rows if row.get("class") == "TypeB"]
    tbox_main = [
        row
        for row in rows
        if row.get("class") == "T_BOX" and row.get("subtype") in {"SCHEMA_UPDATE", "RELAXATION_SET_EXPANSION", "RESTRICTION_SET_CONTRACTION"}
    ]
    tbox_coincidental = [
        row for row in rows if row.get("class") == "T_BOX" and row.get("subtype") == "COINCIDENTAL_SCHEMA_CHANGE"
    ]
    main_candidates = [row for row in rows if _bool_from_csv(row.get("main_score"))]

    by_stratum: dict[str, dict[str, Any]] = {}
    for stratum in sorted({row.get("selection_stratum", "") for row in rows}):
        if not stratum:
            continue
        stratum_rows = [row for row in rows if row.get("selection_stratum") == stratum]
        if stratum.startswith("TypeC"):
            by_stratum[stratum] = _metric_from_rows(
                stratum_rows,
                "typec_judgment",
                {"external_confirmed", "external_by_elimination_ok", "unknown_or_incomplete"},
            )
        elif stratum.startswith("TypeA_DELETE"):
            by_stratum[stratum] = _metric_from_rows(stratum_rows, "typea_judgment", {"delete_ambiguous_ok"})
        elif stratum.startswith("TypeA"):
            by_stratum[stratum] = _metric_from_rows(stratum_rows, "typea_judgment", {"clean_rule_or_format"})
        elif stratum.startswith("TypeB"):
            by_stratum[stratum] = _metric_from_rows(stratum_rows, "typeb_judgment", {"local_confirmed"})
        elif stratum.startswith("TBOX_COINCIDENTAL"):
            by_stratum[stratum] = _metric_from_rows(stratum_rows, "tbox_judgment", {"coincidental_or_weak"})
        elif stratum.startswith("TBOX"):
            by_stratum[stratum] = _metric_from_rows(
                stratum_rows, "tbox_judgment", {"causal_schema_repair", "plausible_schema_update"}
            )

    return {
        "annotation_completeness_rate": _rate(len(complete_rows), total),
        "row_count": total,
        "complete_row_count": len(complete_rows),
        "unannotated_row_count": len(unannotated_rows),
        "label_precision_by_stratum": by_stratum,
        "TypeC_confirmed_external_rate": _metric_from_rows(typec_external, "typec_judgment", {"external_confirmed"}),
        "TypeC_local_missed_rate": _metric_from_rows(typec_external, "typec_judgment", {"local_missed"}),
        "TypeC_unknown_or_incomplete_rate": _metric_from_rows(
            typec_all, "typec_judgment", {"unknown_or_incomplete", "bad_target"}
        ),
        "TypeA_overclaim_rate": _metric_from_rows(
            typea_format,
            "typea_judgment",
            {"overclaimed", "needs_local_evidence", "needs_external_evidence"},
        ),
        "Delete_ambiguity_confirmation_rate": _metric_from_rows(typea_delete, "typea_judgment", {"delete_ambiguous_ok"}),
        "TypeB_local_precision": _metric_from_rows(typeb, "typeb_judgment", {"local_confirmed"}),
        "TypeB_leakage_suspicion_rate": _metric_from_rows(typeb, "typeb_judgment", {"leakage_suspected"}),
        "Tbox_causal_precision": _metric_from_rows(
            tbox_main, "tbox_judgment", {"causal_schema_repair", "plausible_schema_update"}
        ),
        "Tbox_coincidental_rate": _metric_from_rows(tbox_coincidental, "tbox_judgment", {"coincidental_or_weak"}),
        "main_score_keep_rate": _metric_from_rows(main_candidates, "core_recommendation", {"main"}),
        "diagnostic_or_exclude_rate": _metric_from_rows(rows, "core_recommendation", {"diagnostic", "exclude"}),
    }


def write_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    metric_names = [
        "annotation_completeness_rate",
        "TypeC_confirmed_external_rate",
        "TypeC_local_missed_rate",
        "TypeC_unknown_or_incomplete_rate",
        "TypeA_overclaim_rate",
        "Delete_ambiguity_confirmation_rate",
        "TypeB_local_precision",
        "TypeB_leakage_suspicion_rate",
        "Tbox_causal_precision",
        "Tbox_coincidental_rate",
        "main_score_keep_rate",
        "diagnostic_or_exclude_rate",
    ]
    lines = [
        "# Phase D Manual Audit Summary",
        "",
        f"- Rows: {summary['row_count']}",
        f"- Complete rows: {summary['complete_row_count']}",
        f"- Unannotated rows: {summary['unannotated_row_count']}",
        "",
        "| Metric | Rate | Numerator | Denominator |",
        "|---|---:|---:|---:|",
    ]
    for name in metric_names:
        value = summary.get(name)
        if isinstance(value, dict):
            rate = value.get("rate")
            numerator = value.get("numerator")
            denominator = value.get("denominator")
        else:
            rate = value
            numerator = ""
            denominator = summary.get("row_count") if name == "annotation_completeness_rate" else ""
        rate_text = "n/a" if rate is None else f"{rate:.4f}"
        lines.append(f"| {name} | {rate_text} | {numerator} | {denominator} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
