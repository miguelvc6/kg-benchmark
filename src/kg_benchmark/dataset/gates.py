from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

import ijson
from jsonschema import Draft202012Validator

WORK_ARTIFACTS = {
    "popularity": "source/popularity.jsonl",
    "candidates": "source/candidates.jsonl",
    "repairs": "source/repairs.jsonl",
    "world_state": "source/world-state.jsonl",
    "cases": "cases.jsonl",
    "source_provenance": "source-provenance.json",
    "lineage": "lineage.json",
    "dispositions": "audit/dispositions.jsonl",
    "audit_summary": "audit/summary.json",
    "ranking": "selections/ranking.json",
    "eligibility_order": "selections/eligibility-order.jsonl",
    "support_bank": "selections/support-bank.json",
    "group_exclusions": "selections/group-exclusions.json",
    "reserve": "selections/reserve.json",
    "prompt_audit": "selections/prompt-audit.json",
    "per_case_eligibility": "selections/per-case-eligibility.jsonl",
    "prompt_clean_eligibility_order": "selections/prompt-clean-eligibility-order.jsonl",
    "replacements": "selections/replacements.jsonl",
    "main_selection": "selections/main-1200.json",
    "api_selection": "selections/azure-600.json",
}

REPO_ARTIFACTS = {
    "protocol": ("paper/protocol.json", "methodology/protocol.json"),
    "methodology_lock": ("paper/methodology.lock.json", "methodology/methodology.lock.json"),
}

SCHEMA_ARTIFACTS = {
    "schema_dataset_manifest": "dataset-manifest.schema.json",
    "schema_source_provenance": "source-provenance.schema.json",
    "schema_methodology_lock": "methodology-lock.schema.json",
    "schema_popularity": "source-popularity.schema.json",
    "schema_candidate": "source-candidate.schema.json",
    "schema_repair": "source-repair.schema.json",
    "schema_world_state": "world-state-record.schema.json",
    "schema_case": "dataset-case.schema.json",
    "schema_lineage": "artifact-lineage.schema.json",
    "schema_disposition": "audit-disposition.schema.json",
    "schema_audit_summary": "audit-summary.schema.json",
    "schema_ranking": "selection-ranking.schema.json",
    "schema_eligibility_order": "eligibility-order.schema.json",
    "schema_support_bank": "support-bank.schema.json",
    "schema_group_exclusions": "group-exclusions.schema.json",
    "schema_reserve": "reserve-manifest.schema.json",
    "schema_prompt_audit": "selection-prompt-audit.schema.json",
    "schema_per_case_eligibility": "per-case-eligibility.schema.json",
    "schema_replacement": "selection-replacement.schema.json",
    "schema_selection": "selection-manifest.schema.json",
}

CANONICAL_FILES = {
    **WORK_ARTIFACTS,
    **{role: destination for role, (_, destination) in REPO_ARTIFACTS.items()},
    **{role: f"schemas/{filename}" for role, filename in SCHEMA_ARTIFACTS.items()},
}

MAIN_REQUESTED = {"IC-L": 230, "IC-G": 375, "IC-E-elim": 295, "TBOX": 300}
AZURE_REQUESTED = {"IC-L": 115, "IC-G": 188, "IC-E-elim": 147, "TBOX": 150}
RESERVE_REQUESTED = {"IC-L": 276, "IC-G": 450, "IC-E-elim": 354, "TBOX": 360}
TBOX_TARGET = {
    "relaxation_expansions": 130,
    "restriction_contractions": 50,
    "schema_updates": 120,
}


class DatasetGateError(ValueError):
    """Raised when a candidate cannot be promoted as the final paper dataset."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetGateError(f"Invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DatasetGateError(f"JSON artifact must contain an object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetGateError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise DatasetGateError(f"Expected an object at {path}:{line_number}")
            yield value


def _schema(root: Path, role: str) -> dict[str, Any]:
    return _load_json(root / CANONICAL_FILES[role])


def _validate_json(root: Path, artifact_role: str, schema_role: str) -> dict[str, Any]:
    value = _load_json(root / CANONICAL_FILES[artifact_role])
    errors = list(Draft202012Validator(_schema(root, schema_role)).iter_errors(value))
    if errors:
        raise DatasetGateError(f"{artifact_role} fails {schema_role}: {errors[0].message}")
    return value


def _validated_jsonl_rows(
    root: Path,
    artifact_role: str,
    schema_role: str,
) -> Iterable[dict[str, Any]]:
    validator = Draft202012Validator(_schema(root, schema_role))
    for index, row in enumerate(_iter_jsonl(root / CANONICAL_FILES[artifact_role]), 1):
        errors = list(validator.iter_errors(row))
        if errors:
            raise DatasetGateError(
                f"{artifact_role} record {index} fails {schema_role}: {errors[0].message}"
            )
        yield row


def _validate_jsonl(root: Path, artifact_role: str, schema_role: str) -> list[dict[str, Any]]:
    return list(_validated_jsonl_rows(root, artifact_role, schema_role))


def _validate_jsonl_count(root: Path, artifact_role: str, schema_role: str) -> int:
    return sum(1 for _ in _validated_jsonl_rows(root, artifact_role, schema_role))


def _validate_jsonl_unique_values(
    root: Path,
    artifact_role: str,
    schema_role: str,
    field: str,
) -> set[str]:
    values: set[str] = set()
    for row in _validated_jsonl_rows(root, artifact_role, schema_role):
        value = row.get(field)
        if not isinstance(value, str) or not value or value in values:
            raise DatasetGateError(f"{artifact_role} requires unique nonempty {field} values.")
        values.add(value)
    return values


def _unique_map(rows: list[dict[str, Any]], field: str, role: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(field)
        if not isinstance(value, str) or not value or value in result:
            raise DatasetGateError(f"{role} requires unique nonempty {field} values.")
        result[value] = row
    return result


def _artifact_matches(reference: Any, path: Path) -> bool:
    return (
        isinstance(reference, dict)
        and reference.get("sha256") == sha256_file(path)
        and reference.get("size_bytes") == path.stat().st_size
    )


def _canonical_sha256(value: Any) -> str:
    def normalize_number(item: Any) -> int | float:
        if isinstance(item, Decimal):
            return int(item) if item == item.to_integral_value() else float(item)
        raise TypeError(f"Unsupported canonical JSON value: {type(item).__name__}")

    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=normalize_number,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _support_groups(support_bank: dict[str, Any]) -> set[str]:
    return {
        row["group_key"]
        for rows in support_bank["support_sets"].values()
        for row in rows
    }


def _canonical_source_rows(path: Path, role: str) -> Iterable[dict[str, Any]]:
    if role == "popularity":
        with path.open("rb") as handle:
            for qid, payload in ijson.kvitems(handle, ""):
                yield {"qid": qid, "popularity": payload}
        return
    if role == "candidates":
        with path.open("rb") as handle:
            yield from ijson.items(handle, "item")
        return
    if role == "world_state":
        with path.open("rb") as handle:
            for case_id, payload in ijson.kvitems(handle, ""):
                yield {"id": case_id, "world_state": payload}
        return
    raise AssertionError(f"Unsupported canonical source role: {role}")


def _verify_canonical_source_equivalence(root: Path, provenance: dict[str, Any]) -> None:
    for role in ("popularity", "candidates", "world_state"):
        source_rows = _canonical_source_rows(Path(provenance["sources"][role]["path"]), role)
        canonical_path = root / CANONICAL_FILES[role]
        if role in {"popularity", "world_state"}:
            id_field = "qid" if role == "popularity" else "id"
            _verify_keyed_source_equivalence(source_rows, canonical_path, role, id_field)
            continue
        with canonical_path.open("rb") as canonical_handle:
            canonical_rows = ijson.items(canonical_handle, "", multiple_values=True)
            position = 0
            sentinel = object()
            while True:
                source_row = next(source_rows, sentinel)
                canonical_row = next(canonical_rows, sentinel)
                if source_row is sentinel and canonical_row is sentinel:
                    break
                if source_row != canonical_row:
                    raise DatasetGateError(
                        f"Published canonical {role} row {position + 1} differs from its provenance-bound source."
                    )
                position += 1


def _verify_keyed_source_equivalence(
    source_rows: Iterable[dict[str, Any]],
    canonical_path: Path,
    role: str,
    id_field: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix=f"kg-{role}-equivalence-") as temporary:
        database = Path(temporary) / "records.sqlite"
        with sqlite3.connect(database) as connection:
            connection.execute(
                "CREATE TABLE records (record_id TEXT PRIMARY KEY, digest TEXT NOT NULL, matched INTEGER NOT NULL DEFAULT 0)"
            )
            for source_row in source_rows:
                record_id = source_row.get(id_field) if isinstance(source_row, dict) else None
                if not isinstance(record_id, str) or not record_id:
                    raise DatasetGateError(f"Provenance-bound {role} source contains a row without {id_field}.")
                try:
                    connection.execute(
                        "INSERT INTO records(record_id, digest) VALUES (?, ?)",
                        (record_id, _canonical_sha256(source_row)),
                    )
                except sqlite3.IntegrityError as exc:
                    raise DatasetGateError(
                        f"Provenance-bound {role} source contains duplicate {id_field} {record_id}."
                    ) from exc
            connection.commit()
            with canonical_path.open("rb") as canonical_handle:
                for position, canonical_row in enumerate(
                    ijson.items(canonical_handle, "", multiple_values=True), 1
                ):
                    record_id = canonical_row.get(id_field) if isinstance(canonical_row, dict) else None
                    stored = connection.execute(
                        "SELECT digest, matched FROM records WHERE record_id = ?", (record_id,)
                    ).fetchone()
                    if (
                        not isinstance(record_id, str)
                        or stored is None
                        or stored[1] != 0
                        or stored[0] != _canonical_sha256(canonical_row)
                    ):
                        raise DatasetGateError(
                            f"Published canonical {role} row {position} differs from its provenance-bound source."
                        )
                    connection.execute(
                        "UPDATE records SET matched = 1 WHERE record_id = ?", (record_id,)
                    )
            unmatched = connection.execute("SELECT COUNT(*) FROM records WHERE matched = 0").fetchone()[0]
            if unmatched:
                raise DatasetGateError(
                    f"Published canonical {role} is missing {unmatched} provenance-bound source rows."
                )


def _verify_source_provenance(root: Path, *, check_external_files: bool) -> dict[str, Any]:
    provenance = _validate_json(root, "source_provenance", "schema_source_provenance")
    if not check_external_files:
        return provenance
    for role, record in provenance["sources"].items():
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise DatasetGateError(f"Source-provenance artifact {role} no longer matches its recorded bytes.")
    acquisition = provenance["acquisition_config"]
    acquisition_path = Path(acquisition["path"])
    if (
        not acquisition_path.is_file()
        or acquisition_path.stat().st_size != acquisition["bytes"]
        or sha256_file(acquisition_path) != acquisition["sha256"]
        or _load_json(acquisition_path) != acquisition["config"]
    ):
        raise DatasetGateError("Acquisition configuration no longer matches source provenance.")
    cache = provenance["cache_provenance"]
    cache_root = Path(cache["root"])
    actual_files = sorted(path for path in cache_root.rglob("*") if path.is_file()) if cache_root.is_dir() else []
    if [path.relative_to(cache_root).as_posix() for path in actual_files] != [row["path"] for row in cache["files"]]:
        raise DatasetGateError("Acquisition cache inventory changed after provenance was recorded.")
    aggregate = hashlib.sha256()
    total_bytes = 0
    for path, record in zip(actual_files, cache["files"], strict=True):
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != record["bytes"] or digest != record["sha256"]:
            raise DatasetGateError(f"Acquisition cache file changed: {record['path']}")
        total_bytes += size
        aggregate.update(record["path"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\0")
    if (
        len(actual_files) != cache["file_count"]
        or total_bytes != cache["total_bytes"]
        or aggregate.hexdigest() != cache["aggregate_sha256"]
    ):
        raise DatasetGateError("Acquisition cache provenance aggregate does not reproduce.")
    _verify_canonical_source_equivalence(root, provenance)
    return provenance


def _verify_methodology(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _load_json(root / CANONICAL_FILES["protocol"])
    lock = _validate_json(root, "methodology_lock", "schema_methodology_lock")
    if protocol.get("status") != "frozen":
        raise DatasetGateError("Published protocol is not frozen.")
    if lock["protocol_id"] != protocol.get("protocol_id"):
        raise DatasetGateError("Methodology lock protocol_id does not match the published protocol.")
    if lock["files"].get("paper/protocol.json") != sha256_file(root / CANONICAL_FILES["protocol"]):
        raise DatasetGateError("Published protocol SHA-256 does not match the methodology lock.")
    return protocol, lock


def _verify_lineage(root: Path, source_provenance: dict[str, Any]) -> dict[str, Any]:
    from artifact_lineage import verify_bound_lineage_manifest

    lineage_path = root / CANONICAL_FILES["lineage"]
    lineage = _validate_json(root, "lineage", "schema_lineage")
    bound = verify_bound_lineage_manifest(
        lineage_path,
        stage2_path=root / CANONICAL_FILES["repairs"],
        stage3_path=root / CANONICAL_FILES["world_state"],
        stage4_path=root / CANONICAL_FILES["cases"],
        allow_exact_equivalent_stage2_jsonl=True,
    )
    if bound["passed"] is not True:
        raise DatasetGateError("Published Stage 2/3/4 artifacts do not satisfy the bound lineage manifest.")
    if source_provenance not in lineage["source_provenance"]:
        raise DatasetGateError("Lineage manifest does not contain the complete published source provenance.")
    source_bindings = {
        "stage0": "popularity",
        "stage1": "candidates",
        "stage2_json": "repairs",
    }
    for lineage_role, source_role in source_bindings.items():
        if lineage["artifacts"][lineage_role]["sha256"] != source_provenance["sources"][source_role]["sha256"]:
            raise DatasetGateError(f"Lineage {lineage_role} does not match source provenance {source_role}.")
    return lineage


def _verify_audit(
    root: Path,
    case_ids: set[str],
    dispositions: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    summary = _validate_json(root, "audit_summary", "schema_audit_summary")
    disposition_by_id = _unique_map(dispositions, "case_id", "dispositions")
    if set(disposition_by_id) != case_ids:
        raise DatasetGateError("Final dispositions do not exactly and uniquely cover every published case.")
    counts = Counter(row["disposition"] for row in dispositions)
    if summary["counts"]["cases"] != len(case_ids) or summary["counts"]["by_disposition"] != dict(sorted(counts.items())):
        raise DatasetGateError("Audit summary disposition counts do not reproduce from published rows.")
    provenance_paths = {
        "cases": root / CANONICAL_FILES["cases"],
        "world_state": root / CANONICAL_FILES["world_state"],
        "stage2": root / CANONICAL_FILES["repairs"],
        "lineage_manifest": root / CANONICAL_FILES["lineage"],
        "stage4_schema": root / CANONICAL_FILES["schema_case"],
        "dispositions": root / CANONICAL_FILES["dispositions"],
    }
    for role, path in provenance_paths.items():
        if not _artifact_matches(summary["provenance"].get(role), path):
            raise DatasetGateError(f"Audit summary does not bind the published {role} artifact.")
    return summary, {case_id: row["disposition"] for case_id, row in disposition_by_id.items()}


def _verify_population_provenance(root: Path, population: dict[str, Any], name: str) -> None:
    expected = {
        "dataset": "cases",
        "audit": "audit_summary",
        "audit_dispositions": "dispositions",
        "exclusions": "group_exclusions",
        "ranking": "ranking",
        "support_bank": "support_bank",
        "eligibility_order": "eligibility_order",
        "prompt_audit": "prompt_audit",
        "per_case_eligibility": "per_case_eligibility",
    }
    for provenance_role, artifact_role in expected.items():
        if not _artifact_matches(
            population["provenance"].get(provenance_role),
            root / CANONICAL_FILES[artifact_role],
        ):
            raise DatasetGateError(f"{name} selection does not bind the published {provenance_role} artifact.")


def _verify_selection(
    root: Path,
    case_ids: set[str],
    dispositions: dict[str, str],
    protocol: dict[str, Any],
) -> dict[str, int]:
    from kg_benchmark.selection.extensible import (
        _effective_quotas,
        _support_group_keys,
        build_eligibility_order,
        build_support_bank,
        materialize_population,
        reserve_quotas,
        tbox_composition,
    )
    from kg_benchmark.selection.workflow import _prompt_clean_order, _replacements

    ranking = _validate_json(root, "ranking", "schema_ranking")
    if ranking["tbox_weighted_prefix_target"] != TBOX_TARGET:
        raise DatasetGateError("T-box weighted prefix target differs from the paper policy.")
    exclusions = _validate_json(root, "group_exclusions", "schema_group_exclusions")
    eligibility = _validate_jsonl(root, "eligibility_order", "schema_eligibility_order")
    expected_eligibility, records_by_id = build_eligibility_order(
        cases_path=root / CANONICAL_FILES["cases"],
        dispositions_path=root / CANONICAL_FILES["dispositions"],
        seed=ranking["seed"],
        excluded_group_keys=set(exclusions["group_keys"]),
        tbox_targets=ranking["tbox_weighted_prefix_target"],
    )
    if eligibility != expected_eligibility:
        raise DatasetGateError("Published eligibility ordering does not reproduce from cases, dispositions, and exclusions.")
    support = _validate_json(root, "support_bank", "schema_support_bank")
    expected_support = build_support_bank(
        eligibility_rows=expected_eligibility,
        records_by_id=records_by_id,
        capacity_per_locus=support["capacity_per_locus"],
    )
    if support != expected_support:
        raise DatasetGateError("Published support bank does not reproduce from the frozen eligibility order.")
    support_groups = _support_group_keys(support)

    reserve = _validate_json(root, "reserve", "schema_reserve")
    if reserve["requested_quotas"] != RESERVE_REQUESTED:
        raise DatasetGateError("Reserve requested quotas differ from the paper policy.")
    effective_reserve = reserve_quotas(
        requested=RESERVE_REQUESTED,
        eligibility_rows=eligibility,
        support_bank=support,
    )
    expected_reserve = materialize_population(
        name="reserve-1440",
        quotas=effective_reserve,
        eligibility_rows=eligibility,
        support_bank=support,
    )
    if (
        reserve["quotas"] != effective_reserve
        or reserve["case_count"] != expected_reserve["case_count"]
        or reserve["selected_case_ids"] != expected_reserve["selected_case_ids"]
        or reserve["selected_group_keys"] != expected_reserve["selected_group_keys"]
        or reserve["tbox_composition"] != tbox_composition(eligibility, reserve["selected_case_ids"])
    ):
        raise DatasetGateError("Published reserve does not reproduce from the frozen eligibility order.")
    if support_groups & set(reserve["selected_group_keys"]):
        raise DatasetGateError("Reserve overlaps the few-shot support bank.")

    prompt_audit = _validate_json(root, "prompt_audit", "schema_prompt_audit")
    if prompt_audit["reserve_cases"] != reserve["case_count"]:
        raise DatasetGateError("Prompt audit reserve count disagrees with the published reserve.")
    conditions = protocol.get("conditions")
    if not isinstance(conditions, dict):
        raise DatasetGateError("Published protocol has no prompt-condition configuration.")
    regimes = conditions.get("prompt_regimes")
    contexts = conditions.get("context_bundles")
    if not isinstance(regimes, list) or not regimes or not isinstance(contexts, list) or not contexts:
        raise DatasetGateError("Published protocol has invalid prompt regimes or context bundles.")
    expected_prompt_attempts = reserve["case_count"] * 2 * len(regimes) * len(contexts)
    if (
        prompt_audit["rendered_prompts"] > expected_prompt_attempts
        or prompt_audit["deterministically_scanned_prompts"] != prompt_audit["rendered_prompts"]
    ):
        raise DatasetGateError("Prompt audit does not prove complete deterministic scanning of attempted prompts.")
    if (
        set(prompt_audit["failed_case_ids"]) != set(prompt_audit["failure_reasons"])
        or prompt_audit["failed_reserve_cases"] != len(prompt_audit["failed_case_ids"])
        or prompt_audit["prompt_clean_reserve_cases"]
        != reserve["case_count"] - prompt_audit["failed_reserve_cases"]
    ):
        raise DatasetGateError("Prompt audit failure and clean-case counts do not reproduce.")
    per_case = _validate_jsonl(root, "per_case_eligibility", "schema_per_case_eligibility")
    per_case_by_id = _unique_map(per_case, "case_id", "per-case eligibility")
    if set(per_case_by_id) != case_ids:
        raise DatasetGateError("Per-case eligibility does not cover the complete published dataset.")
    for case_id, row in per_case_by_id.items():
        basis = {key: value for key, value in row.items() if key != "eligibility_sha256"}
        if row["eligibility_sha256"] != _canonical_sha256(basis):
            raise DatasetGateError(f"Per-case eligibility digest mismatch for {case_id}.")
        if row["disposition"] != dispositions[case_id]:
            raise DatasetGateError(f"Per-case eligibility disposition mismatch for {case_id}.")

    clean = _validate_jsonl(root, "prompt_clean_eligibility_order", "schema_eligibility_order")
    clean_ids = [row["case_id"] for row in clean]
    eligible_ids = {case_id for case_id, row in per_case_by_id.items() if row["selection_eligible"] is True}
    if len(clean_ids) != len(set(clean_ids)) or set(clean_ids) != eligible_ids:
        raise DatasetGateError("Prompt-clean order and per-case selection eligibility disagree.")
    reserve_ids = set(reserve["selected_case_ids"])
    if not set(prompt_audit["failed_case_ids"]).issubset(reserve_ids):
        raise DatasetGateError("Prompt audit contains failures outside the published reserve.")
    reserve_rows = [row for row in eligibility if row["case_id"] in reserve_ids]
    expected_clean = _prompt_clean_order(
        reserve_rows,
        prompt_audit["failure_reasons"],
        ranking["tbox_weighted_prefix_target"],
    )
    if clean != expected_clean:
        raise DatasetGateError("Prompt-clean order does not reproduce after removing audited failures.")
    if len(clean) != prompt_audit["prompt_clean_reserve_cases"]:
        raise DatasetGateError("Prompt-clean order count disagrees with the prompt audit.")
    if set(prompt_audit["failed_case_ids"]) & set(clean_ids):
        raise DatasetGateError("Prompt-failed cases remain in the clean eligibility order.")

    main = _validate_json(root, "main_selection", "schema_selection")
    azure = _validate_json(root, "api_selection", "schema_selection")
    for name, population in (("main", main), ("azure", azure)):
        _verify_population_provenance(root, population, name)
        if set(population["case_eligibility_sha256"]) != set(population["selected_case_ids"]):
            raise DatasetGateError(f"{name} selection lacks per-case eligibility hashes.")
        for case_id, digest in population["case_eligibility_sha256"].items():
            if digest != per_case_by_id[case_id]["eligibility_sha256"]:
                raise DatasetGateError(f"{name} eligibility digest mismatch for {case_id}.")
    if main["manifest_version"] != 2 or main["requested_quotas"] != MAIN_REQUESTED:
        raise DatasetGateError("Main selection is not the required version-2 paper population.")
    effective_main = _effective_quotas(
        requested=MAIN_REQUESTED,
        eligibility_rows=clean,
        support_bank=support,
    )
    expected_main = materialize_population(
        name="main-1200",
        quotas=effective_main,
        eligibility_rows=clean,
        support_bank=support,
    )
    if (
        main["case_count"] != 1200
        or main["quotas"] != effective_main
        or main["selected_case_ids"] != expected_main["selected_case_ids"]
        or main["selected_group_keys"] != expected_main["selected_group_keys"]
        or main["tbox_composition"] != tbox_composition(clean, main["selected_case_ids"])
    ):
        raise DatasetGateError("Main selection does not reproduce as exactly 1,200 prompt-clean independent cases.")
    main_ids = set(main["selected_case_ids"])
    main_order = [row for row in clean if row["case_id"] in main_ids]
    if azure["manifest_version"] != 2 or azure["requested_quotas"] != AZURE_REQUESTED:
        raise DatasetGateError("Azure selection is not the required version-2 calibration population.")
    effective_azure = _effective_quotas(
        requested=AZURE_REQUESTED,
        eligibility_rows=main_order,
        support_bank=support,
    )
    expected_azure = materialize_population(
        name="azure-600",
        quotas=effective_azure,
        eligibility_rows=main_order,
        support_bank=support,
    )
    if (
        azure["case_count"] != 600
        or azure["quotas"] != effective_azure
        or azure["selected_case_ids"] != expected_azure["selected_case_ids"]
        or azure["selected_group_keys"] != expected_azure["selected_group_keys"]
        or not set(azure["selected_case_ids"]).issubset(main_ids)
    ):
        raise DatasetGateError("Azure selection is not an exact nested 600-case subset of the main population.")
    replacements = _validate_jsonl(root, "replacements", "schema_replacement")
    expected_replacements = _replacements(reserve_rows, set(clean_ids), effective_main)
    if replacements != expected_replacements or prompt_audit["replacement_count"] != len(replacements):
        raise DatasetGateError("Prompt-failure replacements do not reproduce from the frozen reserve order.")
    return {
        "eligible_groups": len(eligibility),
        "reserve_cases": reserve["case_count"],
        "prompt_clean_cases": len(clean),
        "main_cases": main["case_count"],
        "azure_cases": azure["case_count"],
    }


def validate_dataset_semantics(root: Path, *, check_external_sources: bool = False) -> dict[str, Any]:
    root = root.resolve()
    protocol, methodology_lock = _verify_methodology(root)
    source_provenance = _verify_source_provenance(root, check_external_files=check_external_sources)
    acquisition_methodology = source_provenance["acquisition_config"]["config"]["methodology"]
    if (
        acquisition_methodology["freeze_scope_sha256"] != methodology_lock["freeze_scope_sha256"]
        or acquisition_methodology["source_git_revision"] != methodology_lock["source_git_revision"]
        or acquisition_methodology["methodology_lock_sha256"]
        != sha256_file(root / CANONICAL_FILES["methodology_lock"])
    ):
        raise DatasetGateError("Acquisition provenance does not match the published methodology lock.")
    _verify_lineage(root, source_provenance)

    popularity_ids = _validate_jsonl_unique_values(
        root, "popularity", "schema_popularity", "qid"
    )
    candidate_count = _validate_jsonl_count(root, "candidates", "schema_candidate")
    repair_count = _validate_jsonl_count(root, "repairs", "schema_repair")
    world_state_count = _validate_jsonl_count(root, "world_state", "schema_world_state")
    case_ids = _validate_jsonl_unique_values(root, "cases", "schema_case", "id")
    dispositions = _validate_jsonl(root, "dispositions", "schema_disposition")
    _verify_audit(root, case_ids, dispositions)
    disposition_map = {row["case_id"]: row["disposition"] for row in dispositions}
    selection_counts = _verify_selection(root, case_ids, disposition_map, protocol)
    return {
        "valid": True,
        "records": {
            "popularity": len(popularity_ids),
            "candidates": candidate_count,
            "repairs": repair_count,
            "world_state": world_state_count,
            "cases": len(case_ids),
            "dispositions": len(dispositions),
        },
        "selection": selection_counts,
        "external_sources_checked": check_external_sources,
    }
