from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

STRATA = ("IC-L", "IC-G", "IC-E-elim", "TBOX")
DEFAULT_MAIN_QUOTAS = {"IC-L": 230, "IC-G": 375, "IC-E-elim": 295, "TBOX": 300}
DEFAULT_API_QUOTAS = {"IC-L": 115, "IC-G": 188, "IC-E-elim": 147, "TBOX": 150}
A_BOX_SUPPORT_ROLES = (
    "ic_l_rule",
    "ic_l_normalization",
    "ic_g_local",
    "ic_e_elim",
)
T_BOX_SUPPORT_ROLES = (
    "cq_plus",
    "cq_minus_or_replace",
    "no_causal_empty",
    "other_or_family_only",
)


def _stable_rank(seed: int, *parts: Any) -> str:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _classification(record: dict[str, Any]) -> tuple[str, str]:
    classification = record.get("classification") if isinstance(record.get("classification"), dict) else {}
    return str(classification.get("class") or ""), str(classification.get("subtype") or "")


def stratum_for_record(record: dict[str, Any]) -> str | None:
    class_name, subtype = _classification(record)
    track = record.get("track")
    if track == "T_BOX" or class_name == "T_BOX":
        return "TBOX"
    if class_name == "TypeA":
        return "IC-L"
    if class_name == "TypeB":
        return "IC-G"
    if class_name == "TypeC" and subtype == "EXTERNAL_BY_ELIMINATION":
        return "IC-E-elim"
    return None


def group_key_for_record(record: dict[str, Any]) -> str:
    case_id = str(record.get("id") or "")
    qid = str(record.get("qid") or "")
    pid = str(record.get("property") or "")
    if record.get("track") == "T_BOX" or _classification(record)[0] == "T_BOX":
        target = record.get("repair_target") if isinstance(record.get("repair_target"), dict) else {}
        revision = target.get("property_revision_id") or target.get("revision_id")
        if not pid or revision is None:
            raise ValueError(f"T-box case {case_id} lacks property/revision group fields.")
        return f"TBOX|{pid}|{revision}"
    if not qid or not pid:
        raise ValueError(f"A-box case {case_id} lacks qid/property group fields.")
    return f"ABOX|{qid}|{pid}"


def _dispositions(dispositions_path: Path) -> dict[str, str]:
    dispositions: dict[str, str] = {}
    seen: set[str] = set()
    for row in _iter_jsonl(dispositions_path):
        case_id = row.get("case_id") or row.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Every disposition must have a case_id.")
        if case_id in seen:
            raise ValueError(f"Duplicate disposition for {case_id}.")
        seen.add(case_id)
        disposition = row.get("disposition")
        if not isinstance(disposition, str) or not disposition:
            raise ValueError(f"Disposition for {case_id} is missing.")
        dispositions[case_id] = disposition
    return dispositions


def build_eligibility_order(
    *,
    cases_path: Path,
    dispositions_path: Path,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    dispositions = _dispositions(dispositions_path)
    included = {case_id for case_id, disposition in dispositions.items() if disposition == "include"}
    by_group: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    records_by_id: dict[str, dict[str, Any]] = {}
    all_case_ids: set[str] = set()
    for record in _iter_jsonl(cases_path):
        case_id = record.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Every dataset case must have an id.")
        if case_id in all_case_ids:
            raise ValueError(f"Duplicate dataset case id: {case_id}.")
        all_case_ids.add(case_id)
        if case_id not in included:
            continue
        stratum = stratum_for_record(record)
        if stratum is None:
            continue
        group_key = group_key_for_record(record)
        records_by_id[case_id] = record
        by_group[group_key].append((case_id, record))

    missing_dispositions = sorted(all_case_ids - set(dispositions))
    unknown_dispositions = sorted(set(dispositions) - all_case_ids)
    if missing_dispositions or unknown_dispositions:
        raise ValueError(
            "Disposition coverage must exactly equal dataset case IDs: "
            f"missing={len(missing_dispositions)} unknown={len(unknown_dispositions)}"
        )

    rows: list[dict[str, Any]] = []
    for group_key, candidates in by_group.items():
        candidates.sort(key=lambda item: _stable_rank(seed, "case", group_key, item[0]))
        case_id, record = candidates[0]
        stratum = stratum_for_record(record)
        assert stratum is not None
        rows.append(
            {
                "case_id": case_id,
                "group_key": group_key,
                "stratum": stratum,
                "rank": _stable_rank(seed, "group", stratum, group_key),
            }
        )
    rows.sort(key=lambda row: (STRATA.index(row["stratum"]), row["rank"], row["case_id"]))
    for stratum in STRATA:
        for index, row in enumerate((row for row in rows if row["stratum"] == stratum), 1):
            row["stratum_position"] = index
    return rows, records_by_id


def _support_role(record: dict[str, Any], stratum: str) -> str | None:
    _, subtype = _classification(record)
    if stratum == "IC-L":
        if subtype in {"SET_MEMBERSHIP_REJECTION", "SELF_LINK_REJECTION", "TARGET_REQUIRED_CLAIM"}:
            return "ic_l_rule"
        return "ic_l_normalization"
    if stratum == "IC-G":
        return "ic_g_local"
    if stratum == "IC-E-elim":
        return "ic_e_elim"
    gold = record.get("tbox_taxonomy_patch_gold") or record.get("gold")
    if not isinstance(gold, dict):
        return None
    repairs = gold.get("repairs") if isinstance(gold.get("repairs"), list) else []
    codes = {repair.get("taxonomy_code") for repair in repairs if isinstance(repair, dict)}
    if "CQ_PLUS" in codes:
        return "cq_plus"
    if codes & {"CQ_MINUS", "CQ_REPLACE"}:
        return "cq_minus_or_replace"
    if gold.get("schema_decision") == "NO_CAUSAL_SCHEMA_REPAIR" and not repairs:
        return "no_causal_empty"
    return "other_or_family_only"


def build_support_bank(
    *,
    eligibility_rows: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
    capacity_per_locus: int,
) -> dict[str, Any]:
    if capacity_per_locus <= 0 or capacity_per_locus % 4:
        raise ValueError("Support-bank capacity per locus must be a positive multiple of four.")
    role_queues: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligibility_rows:
        record = records_by_id[row["case_id"]]
        role = _support_role(record, row["stratum"])
        if role is not None:
            role_queues[role].append(row)

    def round_robin(roles: tuple[str, ...], capacity: int) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        per_role = capacity // len(roles)
        for role in roles:
            if len(role_queues[role]) < per_role:
                raise ValueError(
                    f"Support role {role} has {len(role_queues[role])} eligible groups; {per_role} required."
                )
        for position in range(per_role):
            for role in roles:
                row = role_queues[role][position]
                selected.append(
                    {
                        "case_id": row["case_id"],
                        "group_key": row["group_key"],
                        "role": role,
                        "visible_example_id": f"example_{len(selected) + 1:06d}",
                    }
                )
        return selected

    abox = round_robin(A_BOX_SUPPORT_ROLES, capacity_per_locus)
    tbox = round_robin(T_BOX_SUPPORT_ROLES, capacity_per_locus)
    return {
        "manifest_type": "few_shot_support_bank",
        "manifest_version": 1,
        "capacity_per_locus": capacity_per_locus,
        "support_sets": {
            "a_box_repair": abox,
            "t_box_repair": tbox,
            "track_diagnosis": [abox[0], tbox[0]],
        },
    }


def _support_group_keys(support_bank: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for task_rows in support_bank.get("support_sets", {}).values():
        for row in task_rows:
            if isinstance(row, dict) and isinstance(row.get("group_key"), str):
                keys.add(row["group_key"])
    return keys


def _largest_remainder(total: int, weights: dict[str, int]) -> dict[str, int]:
    if total < 0 or not weights or any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Largest-remainder allocation requires a non-negative total and positive weights.")
    denominator = sum(weights.values())
    exact = {key: total * weight / denominator for key, weight in weights.items()}
    allocation = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(allocation.values())
    order = sorted(weights, key=lambda key: (-(exact[key] - allocation[key]), key))
    for key in order[:remainder]:
        allocation[key] += 1
    return allocation


def _effective_quotas(
    *,
    requested: dict[str, int],
    eligibility_rows: list[dict[str, Any]],
    support_bank: dict[str, Any],
) -> dict[str, int]:
    if set(requested) != set(STRATA) or any(not isinstance(value, int) or value < 0 for value in requested.values()):
        raise ValueError(f"Quotas must provide non-negative integers for exactly {STRATA}.")
    blocked_groups = _support_group_keys(support_bank)
    available = {
        stratum: sum(
            row["stratum"] == stratum and row["group_key"] not in blocked_groups for row in eligibility_rows
        )
        for stratum in STRATA
    }
    quotas = dict(requested)
    tbox_deficit = max(0, quotas["TBOX"] - available["TBOX"])
    if tbox_deficit:
        quotas["TBOX"] = available["TBOX"]
        additions = _largest_remainder(
            tbox_deficit,
            {stratum: requested[stratum] for stratum in STRATA if stratum != "TBOX"},
        )
        for stratum, addition in additions.items():
            quotas[stratum] += addition
    for stratum, quota in quotas.items():
        if quota > available[stratum]:
            raise ValueError(
                f"Population requests {quota} {stratum} groups after T-box redistribution; "
                f"only {available[stratum]} available."
            )
    return quotas


def materialize_population(
    *,
    name: str,
    quotas: dict[str, int],
    eligibility_rows: list[dict[str, Any]],
    support_bank: dict[str, Any],
    parent_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if set(quotas) != set(STRATA) or any(not isinstance(value, int) or value < 0 for value in quotas.values()):
        raise ValueError(f"Quotas must provide non-negative integers for exactly {STRATA}.")
    blocked_groups = _support_group_keys(support_bank)
    by_stratum = {
        stratum: [
            row for row in eligibility_rows if row["stratum"] == stratum and row["group_key"] not in blocked_groups
        ]
        for stratum in STRATA
    }
    selected: list[dict[str, Any]] = []
    for stratum in STRATA:
        available = len(by_stratum[stratum])
        requested = quotas[stratum]
        if requested > available:
            raise ValueError(f"Population {name} requests {requested} {stratum} groups; only {available} available.")
        selected.extend(by_stratum[stratum][:requested])
    selected_ids = [row["case_id"] for row in selected]
    parent: dict[str, Any] | None = None
    if parent_manifest is not None:
        parent_ids = set(parent_manifest.get("selected_case_ids", []))
        parent_quotas = parent_manifest.get("quotas", {})
        quotas_nondecreasing = all(quotas[stratum] >= int(parent_quotas.get(stratum, 0)) for stratum in STRATA)
        nesting_proven = quotas_nondecreasing and parent_ids.issubset(selected_ids)
        if not nesting_proven:
            raise ValueError(f"Population {name} is not a nested expansion of its parent.")
        parent = {"name": parent_manifest.get("name"), "nesting_proven": True}
    return {
        "manifest_type": "evaluation_population",
        "manifest_version": 1,
        "name": name,
        "quotas": quotas,
        "case_count": len(selected_ids),
        "selected_case_ids": selected_ids,
        "selected_group_keys": [row["group_key"] for row in selected],
        "support_groups_excluded": len(blocked_groups),
        "parent": parent,
    }


def build_selection_artifacts(
    *,
    cases_path: Path,
    dispositions_path: Path,
    output_dir: Path,
    seed: int = 13,
    support_capacity: int = 16,
    main_quotas: dict[str, int] | None = None,
    api_quotas: dict[str, int] | None = None,
) -> dict[str, Any]:
    eligibility_rows, records_by_id = build_eligibility_order(
        cases_path=cases_path,
        dispositions_path=dispositions_path,
        seed=seed,
    )
    support_bank = build_support_bank(
        eligibility_rows=eligibility_rows,
        records_by_id=records_by_id,
        capacity_per_locus=support_capacity,
    )
    effective_main = _effective_quotas(
        requested=dict(main_quotas or DEFAULT_MAIN_QUOTAS),
        eligibility_rows=eligibility_rows,
        support_bank=support_bank,
    )
    main = materialize_population(
        name="main-1200",
        quotas=effective_main,
        eligibility_rows=eligibility_rows,
        support_bank=support_bank,
    )
    main_ids = set(main["selected_case_ids"])
    main_order = [row for row in eligibility_rows if row["case_id"] in main_ids]
    effective_api = _effective_quotas(
        requested=dict(api_quotas or DEFAULT_API_QUOTAS),
        eligibility_rows=main_order,
        support_bank=support_bank,
    )
    api = materialize_population(
        name="azure-600",
        quotas=effective_api,
        eligibility_rows=main_order,
        support_bank=support_bank,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "eligibility-order.jsonl", eligibility_rows)
    _write_json(output_dir / "support-bank.json", support_bank)
    _write_json(output_dir / "main-1200.json", main)
    _write_json(output_dir / "azure-600.json", api)
    return {
        "eligible_groups": len(eligibility_rows),
        "support_groups": len(_support_group_keys(support_bank)),
        "main_cases": main["case_count"],
        "api_cases": api["case_count"],
    }
