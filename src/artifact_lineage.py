"""Streaming lineage validation for Stage 0--4 benchmark artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Iterator

import ijson
from jsonschema import Draft202012Validator

from classifier import lean_repair_target
from kg_benchmark.dataset.release import sha256_file
from lib.utils import iter_jsonl, iter_repairs

LINEAGE_VERSION = 3
RECONCILIATION_MODE = "ordered_filtered_enriched_successor"
STAGE2_PROJECTED_FIELDS = (
    "id",
    "qid",
    "property",
    "track",
    "information_type",
    "violation_context",
    "repair_target",
    "persistence_check",
    "popularity",
)


def canonical_record_bytes(value: Any) -> bytes:
    """Return the stable UTF-8 representation used for record-level lineage."""
    def normalize_number(item: Any) -> Any:
        if isinstance(item, Decimal):
            return int(item) if item == item.to_integral_value() else float(item)
        raise TypeError(f"Unsupported canonical JSON value: {type(item).__name__}")

    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=normalize_number
    ).encode("utf-8")


def canonical_record_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_record_bytes(value)).hexdigest()


def stage2_projection(record: dict[str, Any]) -> dict[str, Any]:
    """Project a Stage 2 row onto fields that the lean Stage 4 artifact must preserve."""
    projected = {key: record[key] for key in STAGE2_PROJECTED_FIELDS if key in record}
    if "repair_target" in projected:
        projected["repair_target"] = lean_repair_target(projected["repair_target"])
    return projected


def stage4_stage2_projection(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in STAGE2_PROJECTED_FIELDS if key in record}


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _artifact(path: Path, *, count: int | None = None, sha256: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256 or sha256_file(path),
    }
    if count is not None:
        result["record_count"] = count
    return result


def _scan_top_level_ids(path: Path, *, json_array: bool) -> tuple[list[str], int]:
    id_prefix = "item.id" if json_array else "id"
    with path.open("rb") as handle:
        ids = [value for value in ijson.items(handle, id_prefix, multiple_values=not json_array)]
    if json_array:
        record_count = len(ids)
    else:
        with path.open("rb") as handle:
            record_count = sum(1 for line in handle if line.strip())
    return ids, record_count


def _representation_from_id_scans(
    json_path: Path,
    jsonl_path: Path,
    *,
    first_left: dict[str, Any] | None,
    first_right: dict[str, Any] | None,
) -> dict[str, Any]:
    json_order, json_count = _scan_top_level_ids(json_path, json_array=True)
    jsonl_order, jsonl_physical_lines = _scan_top_level_ids(jsonl_path, json_array=False)
    jsonl_count = len(jsonl_order)
    duplicate_json = sorted(case_id for case_id, count in Counter(json_order).items() if count > 1)
    duplicate_jsonl = sorted(case_id for case_id, count in Counter(jsonl_order).items() if count > 1)
    json_ids, jsonl_ids = set(json_order), set(jsonl_order)
    checks = {
        "counts_equal": json_count == jsonl_count,
        "ids_complete": len(json_order) == json_count and len(jsonl_order) == jsonl_count,
        "ids_unique": not duplicate_json and not duplicate_jsonl,
        "id_sets_equal": json_ids == jsonl_ids,
        "ordering_equal": json_order == jsonl_order,
        "canonical_record_digests_equal": False,
        "jsonl_one_record_per_line": jsonl_count == jsonl_physical_lines,
    }
    return {
        "passed": False,
        "checks": checks,
        "counts": {
            "json": json_count,
            "jsonl": jsonl_count,
            "jsonl_physical_lines": jsonl_physical_lines,
            "positions_compared": min(json_count, jsonl_count),
        },
        "differences": [
            {
                "position": 0,
                "json_id": first_left.get("id") if isinstance(first_left, dict) else None,
                "jsonl_id": first_right.get("id") if isinstance(first_right, dict) else None,
                "json_digest": canonical_record_sha256(first_left) if first_left is not None else None,
                "jsonl_digest": canonical_record_sha256(first_right) if first_right is not None else None,
            }
        ],
        "duplicate_ids": {"json": duplicate_json[:100], "jsonl": duplicate_jsonl[:100]},
    }


def _stage2_representation_check(json_path: Path, jsonl_path: Path) -> dict[str, Any]:
    json_rows = iter_repairs(json_path)
    jsonl_handle = jsonl_path.open("rb")
    jsonl_rows = ijson.items(jsonl_handle, "", multiple_values=True)
    first_left = next(json_rows, None)
    first_right = next(jsonl_rows, None)
    if first_left != first_right:
        jsonl_handle.close()
        return _representation_from_id_scans(
            json_path, jsonl_path, first_left=first_left, first_right=first_right
        )
    json_rows = chain((first_left,), json_rows) if first_left is not None else iter(())
    jsonl_rows = chain((first_right,), jsonl_rows) if first_right is not None else iter(())
    position = 0
    json_count = 0
    jsonl_count = 0
    json_ids_complete = True
    jsonl_ids_complete = True
    json_ids: set[str] = set()
    jsonl_ids: set[str] = set()
    duplicate_json: set[str] = set()
    duplicate_jsonl: set[str] = set()
    differences: list[dict[str, Any]] = []
    ordering_equal = True
    digests_equal = True
    while True:
        try:
            left = next(json_rows)
        except StopIteration:
            left = None
        try:
            right = next(jsonl_rows)
        except StopIteration:
            right = None
        if left is None and right is None:
            break
        if left is not None:
            json_count += 1
        if right is not None:
            jsonl_count += 1
        left_id = left.get("id") if isinstance(left, dict) else None
        right_id = right.get("id") if isinstance(right, dict) else None
        if isinstance(left_id, str):
            if left_id in json_ids:
                duplicate_json.add(left_id)
            json_ids.add(left_id)
        elif left is not None:
            json_ids_complete = False
        if isinstance(right_id, str):
            if right_id in jsonl_ids:
                duplicate_jsonl.add(right_id)
            jsonl_ids.add(right_id)
        elif right is not None:
            jsonl_ids_complete = False
        if left_id != right_id:
            ordering_equal = False
        records_equal = left == right
        if not records_equal:
            digests_equal = False
            if len(differences) < 100:
                left_digest = canonical_record_sha256(left) if left is not None else None
                right_digest = canonical_record_sha256(right) if right is not None else None
                differences.append(
                    {
                        "position": position,
                        "json_id": left_id,
                        "jsonl_id": right_id,
                        "json_digest": left_digest,
                        "jsonl_digest": right_digest,
                    }
                )
        position += 1
    jsonl_handle.close()
    with jsonl_path.open("rb") as handle:
        jsonl_physical_lines = sum(1 for line in handle if line.strip())
    checks = {
        "counts_equal": json_count == jsonl_count,
        "ids_complete": json_ids_complete and jsonl_ids_complete,
        "ids_unique": not duplicate_json and not duplicate_jsonl,
        "id_sets_equal": json_ids == jsonl_ids,
        "ordering_equal": ordering_equal,
        "canonical_record_digests_equal": digests_equal,
        "jsonl_one_record_per_line": jsonl_count == jsonl_physical_lines,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "json": json_count,
            "jsonl": jsonl_count,
            "jsonl_physical_lines": jsonl_physical_lines,
            "positions_compared": position,
        },
        "differences": differences,
        "duplicate_ids": {
            "json": sorted(duplicate_json)[:100],
            "jsonl": sorted(duplicate_jsonl)[:100],
        },
    }


def _stage2_reconciliation_check(
    authoritative_path: Path,
    precursor_path: Path,
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Prove a declared historical precursor -> compiled Stage 2 transform.

    The restored baseline contains a JSONL precursor and a later JSON array that
    was filtered and enriched before Stage 3 was built.  This check accepts that
    relationship only when every authoritative row occurs in precursor order and
    is byte-canonically identical after removal of explicitly declared top-level
    enrichment fields.  It never rewrites either artifact.
    """
    if policy.get("mode") != RECONCILIATION_MODE:
        raise ValueError(f"Unsupported Stage 2 reconciliation mode: {policy.get('mode')!r}")
    if policy.get("authoritative_artifact") != "stage2_json":
        raise ValueError("Reconciliation must declare stage2_json as authoritative_artifact.")
    if policy.get("precursor_artifact") != "stage2_jsonl":
        raise ValueError("Reconciliation must declare stage2_jsonl as precursor_artifact.")
    allowed_fields = policy.get("allowed_authoritative_only_fields")
    if (
        not isinstance(allowed_fields, list)
        or not allowed_fields
        or not all(isinstance(field, str) and field for field in allowed_fields)
        or len(allowed_fields) != len(set(allowed_fields))
    ):
        raise ValueError("Reconciliation requires unique allowed_authoritative_only_fields.")
    if policy.get("allow_precursor_only_records") is not True:
        raise ValueError("The filtered-successor mode must explicitly allow precursor-only records.")
    allow_multivalue_lines = policy.get("allow_precursor_jsonl_multi_value_lines") is True

    precursor_by_id: dict[str, tuple[int, str, tuple[bool, ...]]] = {}
    precursor_duplicates: list[str] = []
    precursor_missing_ids = 0
    precursor_count = 0
    with precursor_path.open("rb") as handle:
        for position, record in enumerate(ijson.items(handle, "", multiple_values=True)):
            precursor_count += 1
            case_id = record.get("id") if isinstance(record, dict) else None
            if not isinstance(case_id, str) or not case_id:
                precursor_missing_ids += 1
                continue
            if case_id in precursor_by_id:
                if len(precursor_duplicates) < 100:
                    precursor_duplicates.append(case_id)
                continue
            field_presence = tuple(field in record for field in allowed_fields)
            normalized = {key: value for key, value in record.items() if key not in allowed_fields}
            precursor_by_id[case_id] = (
                position,
                canonical_record_sha256(normalized),
                field_presence,
            )

    with precursor_path.open("rb") as handle:
        physical_lines = sum(1 for line in handle if line.strip())

    authoritative_ids: set[str] = set()
    authoritative_duplicates: list[str] = []
    authoritative_missing_ids = 0
    authoritative_count = 0
    missing_from_precursor: list[str] = []
    content_mismatches: list[str] = []
    ordering_differences: list[str] = []
    precursor_declared_field_violations: list[str] = []
    enrichment_presence = Counter()
    last_precursor_position = -1
    for record in iter_repairs(authoritative_path):
        authoritative_count += 1
        case_id = record.get("id") if isinstance(record, dict) else None
        if not isinstance(case_id, str) or not case_id:
            authoritative_missing_ids += 1
            continue
        if case_id in authoritative_ids:
            if len(authoritative_duplicates) < 100:
                authoritative_duplicates.append(case_id)
            continue
        authoritative_ids.add(case_id)
        for field in allowed_fields:
            if field in record:
                enrichment_presence[field] += 1
        source = precursor_by_id.get(case_id)
        if source is None:
            if len(missing_from_precursor) < 100:
                missing_from_precursor.append(case_id)
            continue
        precursor_position, precursor_digest, precursor_field_presence = source
        if precursor_position <= last_precursor_position and len(ordering_differences) < 100:
            ordering_differences.append(case_id)
        last_precursor_position = precursor_position
        if any(precursor_field_presence) and len(precursor_declared_field_violations) < 100:
            precursor_declared_field_violations.append(case_id)
        normalized = {key: value for key, value in record.items() if key not in allowed_fields}
        if canonical_record_sha256(normalized) != precursor_digest and len(content_mismatches) < 100:
            content_mismatches.append(case_id)

    precursor_only_ids = [case_id for case_id in precursor_by_id if case_id not in authoritative_ids]
    precursor_only_digest = hashlib.sha256()
    for case_id in precursor_only_ids:
        precursor_only_digest.update(case_id.encode("utf-8"))
        precursor_only_digest.update(b"\n")
    one_record_per_line = precursor_count == physical_lines
    checks = {
        "ids_complete": not precursor_missing_ids and not authoritative_missing_ids,
        "ids_unique": not precursor_duplicates and not authoritative_duplicates,
        "authoritative_ids_are_precursor_subset": not missing_from_precursor,
        "authoritative_order_is_precursor_subsequence": not ordering_differences,
        "shared_records_equal_after_declared_enrichment_removed": not content_mismatches,
        "declared_enrichment_absent_from_precursor": not precursor_declared_field_violations,
        "precursor_only_records_explicitly_allowed": policy.get("allow_precursor_only_records") is True,
        "precursor_jsonl_format_explained": one_record_per_line or allow_multivalue_lines,
    }
    return {
        "mode": RECONCILIATION_MODE,
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "authoritative": authoritative_count,
            "precursor": precursor_count,
            "precursor_physical_lines": physical_lines,
            "shared": len(authoritative_ids) - len(missing_from_precursor),
            "precursor_only": len(precursor_only_ids),
        },
        "declared_transform": {
            "authoritative_artifact": "stage2_json",
            "precursor_artifact": "stage2_jsonl",
            "allowed_authoritative_only_fields": allowed_fields,
            "allow_precursor_only_records": True,
            "allow_precursor_jsonl_multi_value_lines": allow_multivalue_lines,
        },
        "enrichment_field_presence_counts": dict(sorted(enrichment_presence.items())),
        "precursor_only_ids_sha256": precursor_only_digest.hexdigest(),
        "differences": {
            "authoritative_missing_from_precursor": missing_from_precursor,
            "shared_content_mismatches": content_mismatches,
            "ordering": ordering_differences,
            "declared_fields_present_in_precursor": precursor_declared_field_violations,
            "precursor_only_sample": precursor_only_ids[:100],
        },
        "missing_id_counts": {
            "authoritative": authoritative_missing_ids,
            "precursor": precursor_missing_ids,
        },
        "duplicate_ids": {
            "authoritative": authoritative_duplicates,
            "precursor": precursor_duplicates,
        },
    }


def _representation_diagnostic_from_reconciliation(reconciliation: dict[str, Any]) -> dict[str, Any]:
    """Retain the old equality report while making its expected failure non-authoritative."""
    counts = reconciliation["counts"]
    precursor_only = counts["precursor_only"]
    enrichment_present = any(reconciliation["enrichment_field_presence_counts"].values())
    one_record_per_line = counts["precursor"] == counts["precursor_physical_lines"]
    checks = {
        "counts_equal": counts["authoritative"] == counts["precursor"],
        "ids_complete": reconciliation["checks"]["ids_complete"],
        "ids_unique": reconciliation["checks"]["ids_unique"],
        "id_sets_equal": not precursor_only
        and reconciliation["checks"]["authoritative_ids_are_precursor_subset"],
        "ordering_equal": not precursor_only
        and reconciliation["checks"]["authoritative_order_is_precursor_subsequence"],
        "canonical_record_digests_equal": not enrichment_present
        and reconciliation["checks"]["shared_records_equal_after_declared_enrichment_removed"],
        "jsonl_one_record_per_line": one_record_per_line,
    }
    return {
        "passed": all(checks.values()),
        "role": "diagnostic_only_when_stage2_relationship_mode_is_not_exact",
        "checks": checks,
        "counts": {
            "json": counts["authoritative"],
            "jsonl": counts["precursor"],
            "jsonl_physical_lines": counts["precursor_physical_lines"],
            "positions_compared": counts["shared"],
        },
        "differences": reconciliation["differences"],
        "duplicate_ids": reconciliation["duplicate_ids"],
    }
def _candidate_key(record: dict[str, Any]) -> tuple[Any, ...]:
    context = record.get("violation_context")
    context = context if isinstance(context, dict) else {}
    return (
        record.get("qid"),
        record.get("property"),
        context.get("report_fix_date"),
        context.get("report_revision_old"),
        context.get("report_revision_new"),
    )


def _candidate_source_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("qid"),
        record.get("property_id"),
        record.get("fix_date"),
        record.get("report_revision_old"),
        record.get("report_revision_new"),
    )


def _validate_stage0_stage1(stage0_path: Path, stage1_path: Path, stage2_path: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="kg-lineage-") as temporary:
        database = Path(temporary) / "provenance.sqlite"
        with sqlite3.connect(database) as connection:
            connection.executescript(
                """
                CREATE TABLE popularity (qid TEXT PRIMARY KEY, payload_hash TEXT NOT NULL);
                CREATE TABLE candidates (key TEXT PRIMARY KEY);
                """
            )
            popularity_count = 0
            with stage0_path.open("rb") as handle:
                for qid, payload in ijson.kvitems(handle, ""):
                    if isinstance(qid, str) and isinstance(payload, dict):
                        connection.execute(
                            "INSERT OR IGNORE INTO popularity VALUES (?, ?)",
                            (qid, canonical_record_sha256(payload)),
                        )
                        popularity_count += 1
            candidate_count = 0
            with stage1_path.open("rb") as handle:
                for candidate in ijson.items(handle, "item"):
                    if not isinstance(candidate, dict):
                        continue
                    key = canonical_record_sha256(_candidate_source_key(candidate))
                    connection.execute("INSERT OR IGNORE INTO candidates VALUES (?)", (key,))
                    candidate_count += 1
            missing_popularity: list[str] = []
            popularity_mismatches: list[str] = []
            missing_candidate: list[str] = []
            stage2_count = 0
            if stage2_path.suffix == ".jsonl":
                stage2_handle = stage2_path.open("rb")
                stage2_records: Iterable[dict[str, Any]] = ijson.items(
                    stage2_handle, "", multiple_values=True
                )
            else:
                stage2_handle = None
                stage2_records = iter_repairs(stage2_path)
            for record in stage2_records:
                stage2_count += 1
                qid = record.get("qid")
                popularity_source = (
                    connection.execute(
                        "SELECT payload_hash FROM popularity WHERE qid = ?", (qid,)
                    ).fetchone()
                    if isinstance(qid, str)
                    else None
                )
                if popularity_source is None:
                    if len(missing_popularity) < 100:
                        missing_popularity.append(str(qid))
                elif popularity_source[0] != canonical_record_sha256(record.get("popularity")):
                    if len(popularity_mismatches) < 100:
                        popularity_mismatches.append(str(record.get("id")))
                key = canonical_record_sha256(_candidate_key(record))
                if connection.execute("SELECT 1 FROM candidates WHERE key = ?", (key,)).fetchone() is None:
                    if len(missing_candidate) < 100:
                        missing_candidate.append(str(record.get("id")))
            if stage2_handle is not None:
                stage2_handle.close()
    checks = {
        "stage0_nonempty": popularity_count > 0,
        "stage1_nonempty": candidate_count > 0,
        "stage2_popularity_provenance_complete": not missing_popularity,
        "stage2_popularity_payloads_equal": not popularity_mismatches,
        "stage2_candidate_provenance_complete": not missing_candidate,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {"stage0": popularity_count, "stage1": candidate_count, "stage2": stage2_count},
        "missing_popularity_qids": missing_popularity,
        "popularity_payload_mismatch_case_ids": popularity_mismatches,
        "missing_candidate_case_ids": missing_candidate,
    }


def _world_ids(path: Path) -> Iterator[str]:
    found = False
    root_key = re.compile(rb'^"((?:[^"\\]|\\.)+)"\s*:')
    with path.open("rb") as handle:
        for line in handle:
            match = root_key.match(line)
            if match:
                found = True
                yield json.loads(b'"' + match.group(1) + b'"')
    if found:
        return
    with path.open("rb") as handle:
        for prefix, event, value in ijson.parse(handle):
            if prefix == "" and event == "map_key" and isinstance(value, str):
                yield value


def _validate_stage234_sqlite(stage2_path: Path, stage3_path: Path, stage4_path: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="kg-lineage-") as temporary:
        database = Path(temporary) / "identity.sqlite"
        with sqlite3.connect(database) as connection:
            connection.executescript(
                """
                CREATE TABLE stage2 (id TEXT PRIMARY KEY, position INTEGER NOT NULL, projection_hash TEXT NOT NULL);
                CREATE TABLE stage3 (id TEXT PRIMARY KEY);
                CREATE TABLE stage4 (id TEXT PRIMARY KEY, position INTEGER NOT NULL);
                """
            )
            duplicates = {"stage2": [], "stage3": [], "stage4": []}
            missing_ids = {"stage2": 0, "stage4": 0}
            for position, record in enumerate(iter_repairs(stage2_path)):
                case_id = record.get("id")
                if not isinstance(case_id, str) or not case_id:
                    missing_ids["stage2"] += 1
                    continue
                try:
                    connection.execute(
                        "INSERT INTO stage2 VALUES (?, ?, ?)",
                        (case_id, position, canonical_record_sha256(stage2_projection(record))),
                    )
                except sqlite3.IntegrityError:
                    if len(duplicates["stage2"]) < 100:
                        duplicates["stage2"].append(case_id)
            for case_id in _world_ids(stage3_path):
                try:
                    connection.execute("INSERT INTO stage3 VALUES (?)", (case_id,))
                except sqlite3.IntegrityError:
                    if len(duplicates["stage3"]) < 100:
                        duplicates["stage3"].append(case_id)
            projection_differences: list[str] = []
            ordering_differences: list[str] = []
            for position, record in enumerate(iter_jsonl(stage4_path)):
                case_id = record.get("id")
                if not isinstance(case_id, str) or not case_id:
                    missing_ids["stage4"] += 1
                    continue
                try:
                    connection.execute("INSERT INTO stage4 VALUES (?, ?)", (case_id, position))
                except sqlite3.IntegrityError:
                    if len(duplicates["stage4"]) < 100:
                        duplicates["stage4"].append(case_id)
                source = connection.execute(
                    "SELECT position, projection_hash FROM stage2 WHERE id = ?", (case_id,)
                ).fetchone()
                if source is not None:
                    if source[0] != position and len(ordering_differences) < 100:
                        ordering_differences.append(case_id)
                    if source[1] != canonical_record_sha256(stage4_stage2_projection(record)):
                        if len(projection_differences) < 100:
                            projection_differences.append(case_id)
            counts = {
                name: connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                for name in ("stage2", "stage3", "stage4")
            }
            differences = {
                "stage2_only_vs_stage3": [
                    row[0] for row in connection.execute(
                        "SELECT id FROM stage2 EXCEPT SELECT id FROM stage3 LIMIT 100"
                    )
                ],
                "stage3_only_vs_stage2": [
                    row[0] for row in connection.execute(
                        "SELECT id FROM stage3 EXCEPT SELECT id FROM stage2 LIMIT 100"
                    )
                ],
                "stage2_only_vs_stage4": [
                    row[0] for row in connection.execute(
                        "SELECT id FROM stage2 EXCEPT SELECT id FROM stage4 LIMIT 100"
                    )
                ],
                "stage4_only_vs_stage2": [
                    row[0] for row in connection.execute(
                        "SELECT id FROM stage4 EXCEPT SELECT id FROM stage2 LIMIT 100"
                    )
                ],
                "ordering": ordering_differences,
                "stage2_projection": projection_differences,
            }
    checks = {
        "ids_complete": not any(missing_ids.values()),
        "ids_unique": not any(duplicates.values()),
        "stage2_stage3_ids_equal": not differences["stage2_only_vs_stage3"] and not differences["stage3_only_vs_stage2"],
        "stage2_stage4_ids_equal": not differences["stage2_only_vs_stage4"] and not differences["stage4_only_vs_stage2"],
        "stage2_stage4_order_equal": not ordering_differences and counts["stage2"] == counts["stage4"],
        "stage2_stage4_projection_equal": not projection_differences and counts["stage2"] == counts["stage4"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "counts": counts,
        "missing_id_counts": missing_ids,
        "duplicate_ids": duplicates,
        "differences": differences,
    }


def _validate_stage234(stage2_path: Path, stage3_path: Path, stage4_path: Path) -> dict[str, Any]:
    """Validate identity/order/projection without materializing canonical copies of every large record."""
    stage2_rows = iter_repairs(stage2_path)
    stage4_handle = stage4_path.open("rb")
    stage4_rows = ijson.items(stage4_handle, "", multiple_values=True)
    stage2_ids: set[str] = set()
    stage4_ids: set[str] = set()
    duplicates = {"stage2": [], "stage3": [], "stage4": []}
    missing_ids = {"stage2": 0, "stage4": 0}
    ordering_differences: list[str] = []
    projection_differences: list[str] = []
    stage2_count = 0
    stage4_count = 0
    position = 0
    while True:
        left = next(stage2_rows, None)
        right = next(stage4_rows, None)
        if left is None and right is None:
            break
        left_id = left.get("id") if isinstance(left, dict) else None
        right_id = right.get("id") if isinstance(right, dict) else None
        if left is not None:
            stage2_count += 1
            if not isinstance(left_id, str) or not left_id:
                missing_ids["stage2"] += 1
            elif left_id in stage2_ids:
                if len(duplicates["stage2"]) < 100:
                    duplicates["stage2"].append(left_id)
            else:
                stage2_ids.add(left_id)
        if right is not None:
            stage4_count += 1
            if not isinstance(right_id, str) or not right_id:
                missing_ids["stage4"] += 1
            elif right_id in stage4_ids:
                if len(duplicates["stage4"]) < 100:
                    duplicates["stage4"].append(right_id)
            else:
                stage4_ids.add(right_id)
        if left_id != right_id and len(ordering_differences) < 100:
            ordering_differences.append(f"{left_id}|{right_id}")
        if left is None or right is None or stage2_projection(left) != stage4_stage2_projection(right):
            if len(projection_differences) < 100:
                projection_differences.append(str(left_id or right_id or f"position_{position}"))
        position += 1
    stage4_handle.close()

    stage3_ids: set[str] = set()
    for case_id in _world_ids(stage3_path):
        if case_id in stage3_ids:
            if len(duplicates["stage3"]) < 100:
                duplicates["stage3"].append(case_id)
        else:
            stage3_ids.add(case_id)
    counts = {"stage2": stage2_count, "stage3": len(stage3_ids), "stage4": stage4_count}
    differences = {
        "stage2_only_vs_stage3": sorted(stage2_ids - stage3_ids)[:100],
        "stage3_only_vs_stage2": sorted(stage3_ids - stage2_ids)[:100],
        "stage2_only_vs_stage4": sorted(stage2_ids - stage4_ids)[:100],
        "stage4_only_vs_stage2": sorted(stage4_ids - stage2_ids)[:100],
        "ordering": ordering_differences,
        "stage2_projection": projection_differences,
    }
    checks = {
        "ids_complete": not any(missing_ids.values()),
        "ids_unique": not any(duplicates.values()),
        "stage2_stage3_ids_equal": stage2_ids == stage3_ids,
        "stage2_stage4_ids_equal": stage2_ids == stage4_ids,
        "stage2_stage4_order_equal": not ordering_differences and stage2_count == stage4_count,
        "stage2_stage4_projection_equal": not projection_differences and stage2_count == stage4_count,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "counts": counts,
        "missing_id_counts": missing_ids,
        "duplicate_ids": duplicates,
        "differences": differences,
    }


def validate_lineage(
    *,
    stage0_path: str | Path,
    stage1_path: str | Path,
    stage2_json_path: str | Path,
    stage2_jsonl_path: str | Path,
    stage3_path: str | Path,
    stage4_path: str | Path,
    source_provenance: Iterable[dict[str, Any]] = (),
    reconciliation_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths = {
        "stage0": Path(stage0_path),
        "stage1": Path(stage1_path),
        "stage2_json": Path(stage2_json_path),
        "stage2_jsonl": Path(stage2_jsonl_path),
        "stage3": Path(stage3_path),
        "stage4": Path(stage4_path),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    total_size = sum(path.stat().st_size for path in paths.values())
    provenance_records = list(source_provenance)
    if reconciliation_policy is not None and not provenance_records:
        raise ValueError("Historical Stage 2 reconciliation requires explicit source provenance.")
    if total_size >= 100 * 1024 * 1024:
        with ProcessPoolExecutor(max_workers=min(9, os.cpu_count() or 1)) as executor:
            if reconciliation_policy is None:
                relationship_future = executor.submit(
                    _stage2_representation_check, paths["stage2_json"], paths["stage2_jsonl"]
                )
            else:
                relationship_future = executor.submit(
                    _stage2_reconciliation_check,
                    paths["stage2_json"],
                    paths["stage2_jsonl"],
                    reconciliation_policy,
                )
            provenance_future = executor.submit(
                _validate_stage0_stage1, paths["stage0"], paths["stage1"], paths["stage2_json"]
            )
            # The compiled JSON is the enriched Stage 2 input used for Stage 3/4 construction.
            identity_future = executor.submit(
                _validate_stage234, paths["stage2_json"], paths["stage3"], paths["stage4"]
            )
            hash_futures = {name: executor.submit(sha256_file, path) for name, path in paths.items()}
            relationship = relationship_future.result()
            provenance = provenance_future.result()
            identity = identity_future.result()
            artifact_hashes = {name: future.result() for name, future in hash_futures.items()}
    else:
        relationship = (
            _stage2_representation_check(paths["stage2_json"], paths["stage2_jsonl"])
            if reconciliation_policy is None
            else _stage2_reconciliation_check(
                paths["stage2_json"], paths["stage2_jsonl"], reconciliation_policy
            )
        )
        provenance = _validate_stage0_stage1(paths["stage0"], paths["stage1"], paths["stage2_json"])
        identity = _validate_stage234(paths["stage2_json"], paths["stage3"], paths["stage4"])
        artifact_hashes = {name: sha256_file(path) for name, path in paths.items()}
    if reconciliation_policy is None:
        relationship = {"mode": "exact_representation_equivalence", **relationship}
        representation = relationship
    else:
        expected_hashes = reconciliation_policy.get("expected_artifact_sha256")
        bindings_valid = (
            isinstance(expected_hashes, dict)
            and expected_hashes.get("stage2_json") == artifact_hashes["stage2_json"]
            and expected_hashes.get("stage2_jsonl") == artifact_hashes["stage2_jsonl"]
        )
        relationship["checks"]["declared_artifact_hashes_match"] = bindings_valid
        relationship["passed"] = relationship["passed"] and bindings_valid
        relationship["declared_transform"]["expected_artifact_sha256"] = expected_hashes
        representation = _representation_diagnostic_from_reconciliation(relationship)
    manifest = {
        "manifest_type": "kg_artifact_lineage",
        "manifest_version": LINEAGE_VERSION,
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "git_revision": _git_revision(),
        "artifacts": {
            name: _artifact(
                path,
                sha256=artifact_hashes[name],
                count=(
                    provenance["counts"][name]
                    if name in {"stage0", "stage1"}
                    else representation["counts"]["json"]
                    if name == "stage2_json"
                    else representation["counts"]["jsonl"]
                    if name == "stage2_jsonl"
                    else identity["counts"][name]
                ),
            )
            for name, path in paths.items()
        },
        "source_provenance": provenance_records,
        "validation": {
            "passed": relationship["passed"] and provenance["passed"] and identity["passed"],
            "authoritative_stage2_artifact": "stage2_json",
            "stage2_relationship": relationship,
            "stage2_representation_equivalence": representation,
            "stage0_stage1_provenance": provenance,
            "stage234_identity_and_projection": identity,
        },
    }
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "artifact-lineage.schema.json"
    Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(manifest)
    return manifest


def refresh_lineage_provenance(
    *,
    prior_manifest_path: str | Path,
    stage0_path: str | Path,
    stage1_path: str | Path,
    stage2_json_path: str | Path,
) -> dict[str, Any]:
    """Recompute Stage 0/1 provenance while retaining hash-bound exhaustive checks.

    This is intended for a validator-only provenance rule upgrade. It refuses to
    reuse relationship or identity results unless every artifact still has the
    exact size and SHA-256 recorded by the prior passing subchecks.
    """
    prior_path = Path(prior_manifest_path)
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "artifact-lineage.schema.json"
    errors = list(
        Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).iter_errors(prior)
    )
    if errors:
        raise ValueError(f"Prior lineage manifest is invalid: {errors[0].message}")
    validation = prior.get("validation", {})
    relationship = validation.get("stage2_relationship")
    identity = validation.get("stage234_identity_and_projection")
    if not isinstance(relationship, dict) or relationship.get("passed") is not True:
        raise ValueError("Prior lineage manifest lacks a passing Stage 2 relationship check.")
    if not isinstance(identity, dict) or identity.get("passed") is not True:
        raise ValueError("Prior lineage manifest lacks a passing Stage 2/3/4 identity check.")

    artifacts = prior.get("artifacts", {})
    paths = {
        "stage0": Path(stage0_path),
        "stage1": Path(stage1_path),
        "stage2_json": Path(stage2_json_path),
    }
    for role, path in paths.items():
        entry = artifacts.get(role, {})
        if not path.is_file() or path.stat().st_size != entry.get("size_bytes"):
            raise ValueError(f"{role} no longer matches the prior lineage artifact size.")
    with ProcessPoolExecutor(max_workers=4) as executor:
        hash_futures = {role: executor.submit(sha256_file, path) for role, path in paths.items()}
        provenance_future = executor.submit(
            _validate_stage0_stage1, paths["stage0"], paths["stage1"], paths["stage2_json"]
        )
        observed_hashes = {role: future.result() for role, future in hash_futures.items()}
        provenance = provenance_future.result()
    mismatches = [
        role for role, digest in observed_hashes.items() if artifacts.get(role, {}).get("sha256") != digest
    ]
    if mismatches:
        raise ValueError(f"Artifacts changed since the prior lineage check: {', '.join(mismatches)}")
    for role in ("stage2_jsonl", "stage3", "stage4"):
        entry = artifacts.get(role, {})
        path_value = entry.get("path")
        path = Path(path_value) if isinstance(path_value, str) else Path("missing")
        if (
            not path.is_file()
            or path.stat().st_size != entry.get("size_bytes")
            or sha256_file(path) != entry.get("sha256")
        ):
            raise ValueError(f"{role} changed since the prior exhaustive lineage check.")

    manifest = json.loads(json.dumps(prior))
    manifest["created_at_utc"] = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    manifest["git_revision"] = _git_revision()
    manifest["validation"]["stage0_stage1_provenance"] = provenance
    manifest["validation"]["passed"] = relationship["passed"] and provenance["passed"] and identity["passed"]
    manifest["validation_reuse"] = {
        "mode": "content_bound_subcheck_reuse",
        "prior_manifest_path": str(prior_path.resolve()),
        "prior_manifest_sha256": sha256_file(prior_path),
        "reused_subchecks": ["stage2_relationship", "stage234_identity_and_projection"],
        "recomputed_subchecks": ["stage0_stage1_provenance"],
        "all_artifact_hashes_reverified": True,
    }
    Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(manifest)
    return manifest


def verify_bound_lineage_manifest(
    manifest_path: str | Path,
    *,
    stage2_path: str | Path,
    stage3_path: str | Path,
    stage4_path: str | Path,
    stage2_sha256: str | None = None,
    stage3_sha256: str | None = None,
    stage4_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify hashes before reusing a complete lineage result in another exhaustive gate."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "artifact-lineage.schema.json"
    errors = list(
        Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).iter_errors(manifest)
    )
    artifacts = manifest.get("artifacts", {})
    stage2 = Path(stage2_path)
    stage2_role = "stage2_jsonl" if stage2.suffix == ".jsonl" else "stage2_json"
    checks = {
        "schema_valid": not errors,
        "stage2_is_declared_authoritative_artifact": stage2_role
        == manifest.get("validation", {}).get("authoritative_stage2_artifact", "stage2_json"),
        "stage2_hash_matches": artifacts.get(stage2_role, {}).get("sha256")
        == (stage2_sha256 or sha256_file(stage2)),
        "stage3_hash_matches": artifacts.get("stage3", {}).get("sha256")
        == (stage3_sha256 or sha256_file(stage3_path)),
        "stage4_hash_matches": artifacts.get("stage4", {}).get("sha256")
        == (stage4_sha256 or sha256_file(stage4_path)),
        "complete_lineage_passed": manifest.get("validation", {}).get("passed") is True,
    }
    identity = manifest.get("validation", {}).get("stage234_identity_and_projection")
    relationship = manifest.get("validation", {}).get("stage2_relationship")
    if manifest.get("manifest_version", 0) >= 3:
        checks["stage2_relationship_passed"] = (
            isinstance(relationship, dict) and relationship.get("passed") is True
        )
    else:
        representation = manifest.get("validation", {}).get("stage2_representation_equivalence")
        checks["stage2_relationship_passed"] = (
            isinstance(representation, dict) and representation.get("passed") is True
        )
    checks["identity_result_present"] = isinstance(identity, dict)
    return {
        "passed": all(checks.values()) and bool(identity.get("passed")) if isinstance(identity, dict) else False,
        "checks": checks,
        "identity": identity,
        "manifest_path": str(path.resolve()),
        "manifest_sha256": sha256_file(path),
        "schema_errors": [error.message for error in errors[:20]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Stage 0--4 artifact lineage.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--stage0", required=True)
    validate.add_argument("--stage1", required=True)
    validate.add_argument("--stage2-json", required=True)
    validate.add_argument("--stage2-jsonl", required=True)
    validate.add_argument("--stage3", required=True)
    validate.add_argument("--stage4", required=True)
    validate.add_argument("--source-provenance", help="Optional JSON array of source records.")
    validate.add_argument(
        "--reconciliation-policy",
        help="Explicit JSON transform declaration for a restored Stage 2 precursor.",
    )
    validate.add_argument("--output", required=True)
    refresh = subparsers.add_parser(
        "refresh-provenance",
        help="Recompute upgraded Stage 0/1 checks and reuse only hash-bound passing subchecks.",
    )
    refresh.add_argument("--prior-manifest", required=True)
    refresh.add_argument("--stage0", required=True)
    refresh.add_argument("--stage1", required=True)
    refresh.add_argument("--stage2-json", required=True)
    refresh.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "refresh-provenance":
        manifest = refresh_lineage_provenance(
            prior_manifest_path=args.prior_manifest,
            stage0_path=args.stage0,
            stage1_path=args.stage1,
            stage2_json_path=args.stage2_json,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest["validation"], sort_keys=True))
        return 0 if manifest["validation"]["passed"] else 1
    provenance = []
    if args.source_provenance:
        provenance = json.loads(Path(args.source_provenance).read_text(encoding="utf-8"))
        if not isinstance(provenance, list):
            raise ValueError("--source-provenance must contain a JSON array.")
    reconciliation_policy = None
    if args.reconciliation_policy:
        reconciliation_policy = json.loads(
            Path(args.reconciliation_policy).read_text(encoding="utf-8")
        )
        if not isinstance(reconciliation_policy, dict):
            raise ValueError("--reconciliation-policy must contain a JSON object.")
    manifest = validate_lineage(
        stage0_path=args.stage0,
        stage1_path=args.stage1,
        stage2_json_path=args.stage2_json,
        stage2_jsonl_path=args.stage2_jsonl,
        stage3_path=args.stage3,
        stage4_path=args.stage4,
        source_provenance=provenance,
        reconciliation_policy=reconciliation_policy,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["validation"], sort_keys=True))
    return 0 if manifest["validation"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
