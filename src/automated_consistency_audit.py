"""Exhaustive Stage 4 consistency audit and blinded AI-review packet export."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import fastjsonschema
import ijson
from jsonschema import Draft202012Validator

from artifact_lineage import _validate_stage234, verify_bound_lineage_manifest
from artifact_release import sha256_file
from temporal_audit import audit_rendered_prompts

AUDIT_VERSION = 2
CASE_STATUS_PRIORITY = {"pass": 0, "unsupported": 1, "disagreement": 2, "error": 3}
INTEGRITY_CODES = {
    "duplicate_case_id",
    "invalid_json",
    "schema_invalid",
    "missing_case_id",
    "missing_world_state",
    "context_reference_mismatch",
    "locus_shape_ambiguous",
}
TYPE_A_RULE_SUBTYPES = {
    "FORMAT_NORMALIZATION",
    "FORMAT_VALUE_PRUNING",
    "MULTIPLICITY_NORMALIZATION",
    "REJECTION_FORMAT_INVALID",
    "SELF_LINK_REJECTION",
    "SET_MEMBERSHIP_REJECTION",
    "TARGET_REQUIRED_CLAIM",
}
REVIEW_PACKET_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": True,
    "required": ["review_id", "review_type", "blinded_case_id", "instructions"],
    "properties": {
        "review_id": {"type": "string", "minLength": 1},
        "review_type": {"enum": ["construct", "temporal"]},
        "blinded_case_id": {"type": "string", "minLength": 1},
        "instructions": {"type": "string", "minLength": 1},
    },
}


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit or None, "dirty": dirty}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _iter_jsonl_with_errors(path: Path) -> Iterator[tuple[int, dict[str, Any] | None, str | None, bytes]]:
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                parsed = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                yield line_number, None, str(exc), raw_line
                continue
            if not isinstance(parsed, dict):
                yield line_number, None, "JSONL row is not an object", raw_line
                continue
            yield line_number, parsed, None, raw_line


def _scalars(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for nested in value.values():
            yield from _scalars(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _scalars(nested)
    elif value is not None:
        yield str(value)


def _values(value: Any) -> list[str]:
    return [item.strip() for item in _scalars(value) if item.strip()]


def _field(value: Any, dotted_path: str) -> Any:
    current = value
    for part in dotted_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _status(findings: list[dict[str, Any]]) -> str:
    if not findings:
        return "pass"
    return max((str(item["status"]) for item in findings), key=CASE_STATUS_PRIORITY.__getitem__)


class WorldStateLookup:
    """Read an existing sidecar or build a content-bound cache without loading Stage 3 into memory."""

    def __init__(self, world_state_path: Path, cache_dir: Path):
        self.world_state_path = world_state_path
        self.cache_dir = cache_dir
        self.connection: sqlite3.Connection | None = None
        self.index_path: Path | None = None
        self.index_source = ""

    def __enter__(self) -> WorldStateLookup:
        source_stat = self.world_state_path.stat()
        signature = f"{source_stat.st_size}:{source_stat.st_mtime_ns}"
        sidecar = self.world_state_path.with_suffix(self.world_state_path.suffix + ".sqlite")
        if self._valid_index(sidecar, signature):
            self.index_path = sidecar
            self.index_source = "existing_sidecar"
        else:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_key = hashlib.sha256(str(self.world_state_path.resolve()).encode()).hexdigest()[:16]
            cached = self.cache_dir / f"world_state_{cache_key}.sqlite"
            if not self._valid_index(cached, signature):
                self._build_index(cached, signature)
            self.index_path = cached
            self.index_source = "audit_cache"
        assert self.index_path is not None
        self.connection = sqlite3.connect(f"file:{self.index_path}?mode=ro", uri=True)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    @staticmethod
    def _valid_index(path: Path, signature: str) -> bool:
        if not path.is_file():
            return False
        try:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
                row = connection.execute("SELECT value FROM meta WHERE key='source_signature'").fetchone()
                count = connection.execute("SELECT value FROM meta WHERE key='entry_count'").fetchone()
                return row is not None and row[0] == signature and count is not None
        except sqlite3.Error:
            return False

    def _build_index(self, path: Path, signature: str) -> None:
        temporary = path.with_suffix(".tmp.sqlite")
        temporary.unlink(missing_ok=True)
        with sqlite3.connect(temporary) as connection:
            connection.executescript(
                """
                CREATE TABLE world_state (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                """
            )
            batch: list[tuple[str, str]] = []
            count = 0
            with self.world_state_path.open("rb") as handle:
                for case_id, payload in ijson.kvitems(handle, ""):
                    if not isinstance(case_id, str) or not isinstance(payload, dict):
                        continue
                    batch.append((case_id, json.dumps(payload, ensure_ascii=True, separators=(",", ":"))))
                    if len(batch) >= 1000:
                        connection.executemany("INSERT INTO world_state VALUES (?, ?)", batch)
                        count += len(batch)
                        batch.clear()
            if batch:
                connection.executemany("INSERT INTO world_state VALUES (?, ?)", batch)
                count += len(batch)
            connection.executemany(
                "INSERT INTO meta VALUES (?, ?)",
                [("source_signature", signature), ("entry_count", str(count))],
            )
            connection.commit()
        temporary.replace(path)

    def get(self, case_id: str) -> dict[str, Any] | None:
        assert self.connection is not None
        row = self.connection.execute("SELECT payload FROM world_state WHERE id = ?", (case_id,)).fetchone()
        if row is None or not isinstance(row[0], str):
            return None
        value = json.loads(row[0])
        return value if isinstance(value, dict) else None

    def get_fields(self, case_id: str, fields: tuple[str, ...]) -> dict[str, Any] | None:
        """Use SQLite JSON1 to avoid decoding irrelevant multi-level context in Python."""
        assert self.connection is not None
        expressions = ", ".join(f"json_extract(payload, '$.{field}')" for field in fields)
        row = self.connection.execute(
            f"SELECT {expressions} FROM world_state WHERE id = ?",  # noqa: S608 - fields are internal constants
            (case_id,),
        ).fetchone()
        if row is None:
            return None
        result: dict[str, Any] = {}
        for field, raw_value in zip(fields, row, strict=True):
            if raw_value is None:
                continue
            if isinstance(raw_value, str):
                try:
                    result[field] = json.loads(raw_value)
                except json.JSONDecodeError:
                    result[field] = raw_value
            else:
                result[field] = raw_value
        return result

    def count(self) -> int:
        assert self.connection is not None
        row = self.connection.execute("SELECT value FROM meta WHERE key='entry_count'").fetchone()
        return int(row[0]) if row else 0


def _derive_locus(record: dict[str, Any], world_state: dict[str, Any]) -> tuple[str | None, str]:
    target = record.get("repair_target")
    target = target if isinstance(target, dict) else {}
    abox_shape = any(key in target for key in ("action", "old_value", "new_value", "revision_id"))
    tbox_shape = any(key in target for key in ("property_revision_id", "property_revision_prev", "constraint_delta"))
    tbox_shape = tbox_shape or isinstance(world_state.get("constraint_change_context"), dict)
    if abox_shape == tbox_shape:
        return None, "locus_shape_ambiguous"
    return ("A_BOX" if abox_shape else "T_BOX"), "shape_reconstruction"


def _constraint_values(world_state: dict[str, Any], constraint_qid: str, qualifier_pid: str) -> list[str]:
    constraints = _field(world_state, "L4_constraints.constraints")
    if not isinstance(constraints, list):
        return []
    values: list[str] = []
    for constraint in constraints:
        if _field(constraint, "constraint_type.qid") != constraint_qid:
            continue
        for qualifier in constraint.get("qualifiers", []):
            if isinstance(qualifier, dict) and qualifier.get("property_id") == qualifier_pid:
                for value in qualifier.get("values", []):
                    if isinstance(value, dict):
                        values.extend(_values(value.get("raw") if value.get("raw") is not None else value.get("qid")))
    return values


def _format_regexes(world_state: dict[str, Any]) -> list[str]:
    return _constraint_values(world_state, "Q21502404", "P1793")


def _has_constraint(world_state: dict[str, Any], constraint_qid: str) -> bool:
    constraints = _field(world_state, "L4_constraints.constraints")
    return isinstance(constraints, list) and any(
        _field(constraint, "constraint_type.qid") == constraint_qid for constraint in constraints
    )


def _regex_match_profile(values: Iterable[str], patterns: Iterable[str]) -> list[bool]:
    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error:
            continue
    return [all(regex.fullmatch(value) is not None for value in values) for regex in compiled]


def _matches_any_regex(values: Iterable[str], patterns: Iterable[str]) -> bool | None:
    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error:
            continue
    if not compiled:
        return None
    return all(any(regex.fullmatch(value) for regex in compiled) for value in values)


def _type_a_rule_result(record: dict[str, Any], world_state: dict[str, Any]) -> tuple[bool | None, str]:
    subtype = str(_field(record, "classification.subtype") or "")
    if subtype not in TYPE_A_RULE_SUBTYPES:
        return None, "unsupported_type_a_subtype"
    target = record.get("repair_target")
    target = target if isinstance(target, dict) else {}
    old_values = _values(target.get("old_value"))
    new_values = _values(target.get("new_value") if target.get("new_value") is not None else target.get("value"))
    removed = set(old_values) - set(new_values)
    if subtype == "SELF_LINK_REJECTION":
        qid = str(record.get("qid") or "")
        return bool(qid and qid in removed and qid not in new_values), "self_link_delta"
    if subtype == "SET_MEMBERSHIP_REJECTION":
        allowed = set(_constraint_values(world_state, "Q21510859", "P2305"))
        if not allowed:
            return None, "missing_one_of_constraint"
        return bool(removed and removed.isdisjoint(allowed) and set(new_values).issubset(allowed)), "one_of_replay"
    if subtype == "TARGET_REQUIRED_CLAIM":
        qid = str(record.get("qid") or "")
        return bool(qid and qid in new_values and qid not in old_values), "target_required_delta"
    if subtype == "MULTIPLICITY_NORMALIZATION":
        return set(old_values) == set(new_values) and old_values != new_values, "multiplicity_delta"
    patterns = _format_regexes(world_state)
    old_match = _matches_any_regex(old_values, patterns)
    new_match = _matches_any_regex(new_values, patterns)
    if old_match is None or new_match is None:
        return None, "missing_or_invalid_format_constraint"
    if subtype == "REJECTION_FORMAT_INVALID":
        return bool(removed and not old_match), "format_rejection_replay"
    return bool(old_values and new_values and not old_match and new_match), "format_replay"


def _normalized_text_tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    for scalar in _scalars(value):
        normalized = " ".join(scalar.casefold().split())
        if normalized:
            tokens.add(normalized)
    return tokens


def _masked_local_evidence(record: dict[str, Any], world_state: dict[str, Any]) -> dict[str, Any]:
    target_property = str(record.get("property") or "")
    target = record.get("repair_target")
    target = target if isinstance(target, dict) else {}
    target_ids = set(
        _values(target.get("new_value") if target.get("new_value") is not None else target.get("value"))
    )
    l1 = _field(world_state, "L1_ego_node.properties")
    other_properties = {
        key: value
        for key, value in (l1.items() if isinstance(l1, dict) else [])
        if str(key) != target_property
    }
    l2 = world_state.get("L2_labels")
    l2 = l2 if isinstance(l2, dict) else {}
    l2_entities = l2.get("entities")
    masked_l2 = dict(l2)
    if isinstance(l2_entities, dict):
        masked_l2["entities"] = {key: value for key, value in l2_entities.items() if key not in target_ids}
    l3 = world_state.get("L3_neighborhood")
    l3 = l3 if isinstance(l3, dict) else {}
    masked_l3 = {}
    for key, value in l3.items():
        if isinstance(value, list):
            masked_l3[key] = [
                item
                for item in value
                if not isinstance(item, dict)
                or str(item.get("target_qid") or item.get("source_qid") or "") not in target_ids
            ]
        else:
            masked_l3[key] = value
    return {
        "l1_other_properties": other_properties,
        "l2_labels": masked_l2,
        "l3_neighborhood": masked_l3,
    }


def _compact_review_value(value: Any, *, list_limit: int = 40, text_limit: int = 600) -> Any:
    """Bound review-packet size while recording every truncation explicitly."""
    if isinstance(value, str):
        if len(value) <= text_limit:
            return value
        return {"text_prefix": value[:text_limit], "original_characters": len(value), "truncated": True}
    if isinstance(value, list):
        compacted = [_compact_review_value(item, list_limit=list_limit, text_limit=text_limit) for item in value[:list_limit]]
        if len(value) > list_limit:
            compacted.append({"omitted_items": len(value) - list_limit, "truncated": True})
        return compacted
    if isinstance(value, dict):
        return {
            str(key): _compact_review_value(item, list_limit=list_limit, text_limit=text_limit)
            for key, item in value.items()
        }
    return value


def _local_support(record: dict[str, Any], world_state: dict[str, Any]) -> tuple[bool, list[str]]:
    target = record.get("repair_target")
    target = target if isinstance(target, dict) else {}
    truth = _normalized_text_tokens(
        target.get("new_value") if target.get("new_value") is not None else target.get("value")
    )
    truth.update(_normalized_text_tokens(target.get("new_value_labels_en")))
    local = _normalized_text_tokens(_masked_local_evidence(record, world_state))
    matches = sorted(truth & local)
    return bool(matches), matches[:20]


def _signature_hash_valid(signature: Any) -> bool:
    if not isinstance(signature, dict):
        return False
    raw = signature.get("signature_raw")
    expected = signature.get("hash")
    if not isinstance(raw, str) or not isinstance(expected, str):
        return False
    return hashlib.sha1(raw.encode()).hexdigest() == expected


def _changed_constraint_types(before: Any, after: Any) -> set[str]:
    def present_types(value: Any) -> set[str]:
        result: set[str] = set()
        if not isinstance(value, dict) or not isinstance(value.get("signature"), list):
            return result
        for constraint in value["signature"]:
            if isinstance(constraint, dict) and isinstance(constraint.get("constraint_qid"), str):
                result.add(constraint["constraint_qid"])
        return result

    return present_types(before) ^ present_types(after)


def _changed_constraint_entries(before: Any, after: Any) -> set[str]:
    def by_qid(value: Any) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        if not isinstance(value, dict) or not isinstance(value.get("signature"), list):
            return result
        for constraint in value["signature"]:
            if isinstance(constraint, dict) and isinstance(constraint.get("constraint_qid"), str):
                result.setdefault(constraint["constraint_qid"], []).append(
                    json.dumps(
                        constraint, ensure_ascii=True, sort_keys=True, separators=(",", ":")
                    )
                )
        for entries in result.values():
            entries.sort()
        return result

    left, right = by_qid(before), by_qid(after)
    return {qid for qid in left.keys() | right.keys() if left.get(qid) != right.get(qid)}


def _case_findings(record: dict[str, Any], world_state: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    def add(status: str, code: str, detail: Any = None) -> None:
        finding = {"status": status, "code": code}
        if detail is not None:
            finding["detail"] = detail
        findings.append(finding)

    reconstructed, basis = _derive_locus(record, world_state)
    if reconstructed is None:
        add("error", basis)
        return findings
    observed_track = record.get("track")
    target_kind = _field(record, "repair_target.kind")
    classification_class = _field(record, "classification.class")
    class_locus = "T_BOX" if classification_class == "T_BOX" else "A_BOX" if classification_class in {"TypeA", "TypeB", "TypeC"} else None
    for field, observed in (("track", observed_track), ("repair_target.kind", target_kind), ("classification.class", class_locus)):
        if observed != reconstructed:
            add("disagreement", "locus_disagreement", {"field": field, "observed": observed, "reconstructed": reconstructed})

    if reconstructed == "T_BOX":
        signatures = _field(world_state, "constraint_change_context.signatures")
        if not isinstance(signatures, dict):
            add("error", "missing_tbox_signatures")
            return findings
        before, after = signatures.get("before"), signatures.get("after")
        before_valid = _signature_hash_valid(before)
        after_valid = _signature_hash_valid(after)
        if not before_valid:
            add("unsupported", "missing_tbox_history_signature_before")
        if not after_valid:
            add("error", "invalid_tbox_signature_hash_after")
            return findings
        changed = _changed_constraint_types(before, after)
        changed_entries = _changed_constraint_entries(before, after)
        declared = set(_values(_field(record, "repair_target.constraint_delta.changed_constraint_types")))
        if changed != declared:
            add("disagreement", "tbox_changed_constraint_disagreement", {"declared": sorted(declared), "reconstructed": sorted(changed)})
        for side, signature in (("before", before), ("after", after)):
            if side == "before" and not before_valid:
                continue
            expected = _field(record, f"repair_target.constraint_delta.hash_{side}")
            observed = signature.get("hash") if isinstance(signature, dict) else None
            if expected != observed:
                add("disagreement", "tbox_signature_binding_disagreement", {"side": side})
        subtype = str(_field(record, "classification.subtype") or "")
        mapped_constraint = _field(record, "classification.decision_constraint_type_qid")
        if (
            isinstance(mapped_constraint, str)
            and mapped_constraint
            and mapped_constraint not in changed_entries
            and not bool(_field(record, "classification.diagnostics.tbox_diff_summary.target_constraint_is_related_family"))
            and subtype not in {"COINCIDENTAL_SCHEMA_CHANGE", "UNKNOWN_TBOX_CAUSALITY"}
        ):
            add(
                "disagreement",
                "tbox_violation_constraint_disagreement",
                {"mapped_violation_constraint": mapped_constraint, "changed_constraint_types": sorted(declared)},
            )
        if subtype == "COINCIDENTAL_SCHEMA_CHANGE" and declared:
            add("unsupported", "tbox_causality_requires_policy_replay")
        before_hash = before.get("hash") if isinstance(before, dict) else None
        after_hash = after.get("hash") if isinstance(after, dict) else None
        if before_hash == after_hash:
            add("disagreement", "tbox_has_no_semantic_constraint_change")
        return findings

    classification = str(classification_class or "")
    if classification == "TypeA":
        result, rule = _type_a_rule_result(record, world_state)
        if result is None:
            add("unsupported", rule)
        elif not result:
            add("disagreement", "type_a_rule_replay_failed", {"rule": rule})
    elif classification == "TypeB":
        supported, matches = _local_support(record, world_state)
        subtype = str(_field(record, "classification.subtype") or "")
        if subtype == "LOCAL_TEXT_DERIVED":
            add("unsupported", "derived_local_evidence_requires_semantic_review", {"exact_matches": matches})
        elif not supported:
            add("disagreement", "type_b_local_support_not_reproduced")
    elif classification == "TypeC":
        supported, matches = _local_support(record, world_state)
        if supported:
            add("disagreement", "type_c_has_visible_local_support", {"matches": matches})
        subtype = str(_field(record, "classification.subtype") or "")
        if subtype.startswith("UNKNOWN_"):
            add("unsupported", "unknown_type_c_remains_diagnostic")
    else:
        add("error", "unknown_classification_class", classification_class)
    target = record.get("repair_target") if isinstance(record.get("repair_target"), dict) else {}
    final_values = _values(target.get("new_value") if target.get("new_value") is not None else target.get("value"))
    if _has_constraint(world_state, "Q19474404") and len(set(final_values)) > 1:
        add(
            "disagreement",
            "cardinality_constraint_conflict",
            {"constraint": "Q19474404", "unique_final_values": len(set(final_values))},
        )
    patterns = _format_regexes(world_state)
    profile = _regex_match_profile(final_values, patterns)
    if len(profile) > 1 and any(profile) and not all(profile):
        add(
            "disagreement",
            "format_rule_contradiction",
            {"matching_rules": sum(profile), "format_rules": len(profile)},
        )
    return findings


def _redact(value: Any, case_id: str, blind_id: str) -> Any:
    if isinstance(value, str):
        return value.replace(case_id, blind_id)
    if isinstance(value, list):
        return [_redact(item, case_id, blind_id) for item in value]
    if isinstance(value, dict):
        return {_redact(key, case_id, blind_id): _redact(item, case_id, blind_id) for key, item in value.items()}
    return value


def _target_without_labels(record: dict[str, Any]) -> dict[str, Any]:
    target = record.get("repair_target")
    if not isinstance(target, dict):
        return {}
    return {key: value for key, value in target.items() if key not in {"kind", "author"}}


def _construct_packet(case_id: str, blind_id: str, record: dict[str, Any], world_state: dict[str, Any]) -> dict[str, Any]:
    locus_view = {
        "qid": record.get("qid"),
        "property": record.get("property"),
        "violation_context": record.get("violation_context"),
        "constraints": world_state.get("L4_constraints"),
        "constraint_change_context": world_state.get("constraint_change_context"),
    }
    derived, _ = _derive_locus(record, world_state)
    task_view: dict[str, Any]
    if derived == "T_BOX":
        task_view = {
            "task": "tbox_validity",
            "repair_target": _target_without_labels(record),
            "violation_context": record.get("violation_context"),
            "constraint_change_context": world_state.get("constraint_change_context"),
            "constraints": world_state.get("L4_constraints"),
        }
    else:
        task_view = {
            "task": "evidence_sufficiency",
            "repair_target": _target_without_labels(record),
            "violation_context": record.get("violation_context"),
            "masked_local_evidence": _masked_local_evidence(record, world_state),
        }
    packet = {
        "review_id": blind_id,
        "review_type": "construct",
        "blinded_case_id": blind_id,
        "locus_view": locus_view,
        "task_view": task_view,
        "instructions": "Find possible label, evidence, causality, ambiguity, or alternative-repair errors. This is error discovery, not ground-truth annotation.",
    }
    return _compact_review_value(_redact(packet, case_id, blind_id))


def _read_sample_ids(path: Path, limit: int) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    case_ids = [str(row.get("case_id") or "").strip() for row in rows]
    if any(not value for value in case_ids) or len(case_ids) != len(set(case_ids)):
        raise ValueError("Construct sample must contain unique non-empty case_id values.")
    if limit < 1 or limit > len(case_ids):
        raise ValueError(f"construct-review-size must be between 1 and {len(case_ids)}")
    return case_ids[:limit]


def _temporal_packets(
    temporal_report: dict[str, Any], rendered_prompts_path: Path, records: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows = []
    with rendered_prompts_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    packets: list[dict[str, Any]] = []
    private_map: dict[str, str] = {}
    for index, sampled in enumerate(temporal_report.get("manual_review_sample", []), start=1):
        row_number = int(sampled["row"])
        prompt = rows[row_number - 1]
        case_id = str(prompt.get("case_id") or "")
        record = records[case_id]
        blind_id = f"temporal_{index:06d}"
        private_map[blind_id] = case_id
        target = record.get("repair_target") if isinstance(record.get("repair_target"), dict) else {}
        hidden_reference = {
            "new_value": target.get("new_value"),
            "value": target.get("value"),
            "new_value_labels_en": target.get("new_value_labels_en"),
            "new_value_descriptions_en": target.get("new_value_descriptions_en"),
            "current_value_2026": _field(record, "persistence_check.current_value_2026"),
            "constraint_delta": target.get("constraint_delta"),
        }
        packet = {
            "review_id": blind_id,
            "review_type": "temporal",
            "blinded_case_id": blind_id,
            "task": prompt.get("task"),
            "context_bundle": prompt.get("context_bundle"),
            "system_prompt": prompt.get("system_prompt"),
            "user_prompt": prompt.get("user_prompt"),
            "hidden_reference": hidden_reference,
            "instructions": "Flag direct, paraphrased, alias-based, or indirect disclosure of post-repair-only facts. Distinguish expected rule-derived visibility.",
        }
        packets.append(_redact(packet, case_id, blind_id))
    return packets, private_map


def run_audit(
    *,
    classified_benchmark_path: str | Path,
    world_state_path: str | Path,
    stage4_schema_path: str | Path,
    construct_sample_path: str | Path,
    rendered_prompts_path: str | Path,
    render_summary_path: str | Path,
    output_dir: str | Path,
    construct_review_size: int = 450,
    temporal_review_size: int = 50,
    seed: int = 13,
    stage2_path: str | Path | None = None,
    lineage_manifest_path: str | Path | None = None,
    cache_dir: str | Path = ".cache/automated_audit",
) -> dict[str, Any]:
    starting_git_state = _git_state()
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Audit output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    stage4_path = Path(classified_benchmark_path)
    world_path = Path(world_state_path)
    schema_path = Path(stage4_schema_path)
    construct_path = Path(construct_sample_path)
    prompts_path = Path(rendered_prompts_path)
    render_path = Path(render_summary_path)
    sample_ids = _read_sample_ids(construct_path, construct_review_size)
    sample_id_set = set(sample_ids)
    prompt_case_ids: set[str] = set()
    with prompts_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                prompt = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid rendered prompt JSON at line {line_number}") from exc
            if isinstance(prompt, dict) and isinstance(prompt.get("case_id"), str):
                prompt_case_ids.add(prompt["case_id"])
    capture_ids = sample_id_set | prompt_case_ids

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    fast_validator = fastjsonschema.compile(schema)
    stage4_digest = hashlib.sha256()
    seen_ids: set[str] = set()
    selected_records: dict[str, dict[str, Any]] = {}
    selected_world: dict[str, dict[str, Any]] = {}
    tbox_world_cache: dict[str, dict[str, Any]] = {}
    finding_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    subtype_counts: Counter[str] = Counter()
    track_counts: Counter[str] = Counter()
    rows_total = 0
    missing_id_rows = 0
    findings_path = output / "deterministic_findings.jsonl"
    statuses_path = output / "deterministic_case_status.jsonl"
    stage2_sha256: str | None = None
    world_state_sha256: str | None = None

    with WorldStateLookup(world_path, Path(cache_dir)) as world_lookup:
        with findings_path.open("w", encoding="utf-8") as findings_handle, statuses_path.open(
            "w", encoding="utf-8"
        ) as statuses_handle:
            for line_number, record, parse_error, raw_line in _iter_jsonl_with_errors(stage4_path):
                stage4_digest.update(raw_line)
                rows_total += 1
                if parse_error is not None or record is None:
                    finding = {
                        "line": line_number,
                        "case_id": None,
                        "status": "error",
                        "code": "invalid_json",
                        "detail": parse_error,
                    }
                    findings_handle.write(json.dumps(finding, ensure_ascii=True) + "\n")
                    finding_counts["invalid_json"] += 1
                    status_counts["error"] += 1
                    continue
                case_id = record.get("id")
                if not isinstance(case_id, str) or not case_id:
                    missing_id_rows += 1
                    case_id = f"__line_{line_number}"
                    case_findings = [{"status": "error", "code": "missing_case_id"}]
                else:
                    case_findings = []
                    if case_id in seen_ids:
                        case_findings.append({"status": "error", "code": "duplicate_case_id"})
                        case_id = f"{case_id}#duplicate#line_{line_number}"
                    seen_ids.add(case_id)
                schema_error: fastjsonschema.JsonSchemaException | None = None
                try:
                    fast_validator(record)
                except fastjsonschema.JsonSchemaException as exc:
                    schema_error = exc
                if schema_error is not None:
                    case_findings.append(
                        {
                            "status": "error",
                            "code": "schema_invalid",
                            "detail": [{"path": schema_error.path, "message": schema_error.message}],
                        }
                    )
                context_id = _field(record, "context_ref.world_state_id")
                if context_id != record.get("id"):
                    case_findings.append(
                        {
                            "status": "error",
                            "code": "context_reference_mismatch",
                            "detail": {"context_id": context_id},
                        }
                    )
                raw_case_id = str(record.get("id") or "")
                target = record.get("repair_target")
                target = target if isinstance(target, dict) else {}
                tbox_hint = any(
                    key in target for key in ("property_revision_id", "property_revision_prev", "constraint_delta")
                )
                capture_full = raw_case_id in capture_ids
                if tbox_hint and not capture_full:
                    revision = target.get("property_revision_id")
                    cache_key = f"{record.get('property')}|{revision}"
                    world_state = tbox_world_cache.get(cache_key)
                    if world_state is None:
                        world_state = world_lookup.get_fields(raw_case_id, ("constraint_change_context",))
                        if world_state is not None:
                            tbox_world_cache[cache_key] = world_state
                elif _field(record, "classification.class") == "TypeA" and not capture_full:
                    world_state = world_lookup.get_fields(raw_case_id, ("L4_constraints",))
                else:
                    world_state = world_lookup.get(raw_case_id)
                if world_state is None:
                    case_findings.append({"status": "error", "code": "missing_world_state"})
                else:
                    case_findings.extend(_case_findings(record, world_state))
                case_status = _status(case_findings)
                status_counts[case_status] += 1
                track_counts[str(record.get("track") or "missing")] += 1
                class_counts[str(_field(record, "classification.class") or "missing")] += 1
                subtype_counts[str(_field(record, "classification.subtype") or "missing")] += 1
                statuses_handle.write(
                    json.dumps(
                        {
                            "case_id": case_id,
                            "status": case_status,
                            "finding_codes": [item["code"] for item in case_findings],
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                    + "\n"
                )
                for finding in case_findings:
                    finding_counts[str(finding["code"])] += 1
                    findings_handle.write(
                        json.dumps(
                            {"line": line_number, "case_id": record.get("id"), **finding},
                            ensure_ascii=True,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                if record.get("id") in capture_ids:
                    selected_records[str(record["id"])] = record
                    if world_state is not None:
                        selected_world[str(record["id"])] = world_state

        missing_sample = sorted(sample_id_set - selected_records.keys())
        if missing_sample:
            raise ValueError(f"Stage 4 is missing {len(missing_sample)} construct sample cases.")

        temporal_report = audit_rendered_prompts(
            rendered_prompts_path=prompts_path,
            classified_benchmark_path=stage4_path,
            sample_size=temporal_review_size,
            seed=seed,
        )
        _write_json(output / "temporal_audit.json", temporal_report)
        temporal_case_ids = {str(row["case_id"]) for row in temporal_report.get("manual_review_sample", [])}
        for case_id in temporal_case_ids:
            if case_id not in selected_records:
                raise ValueError(f"Temporal sample case is missing from Stage 4: {case_id}")

        construct_packets: list[dict[str, Any]] = []
        private_map: dict[str, str] = {}
        for index, case_id in enumerate(sample_ids, start=1):
            blind_id = f"construct_{index:06d}"
            private_map[blind_id] = case_id
            construct_packets.append(
                _construct_packet(case_id, blind_id, selected_records[case_id], selected_world[case_id])
            )
        temporal_packets, temporal_map = _temporal_packets(temporal_report, prompts_path, selected_records)
        private_map.update(temporal_map)

        construct_packets_path = output / "construct_review_packets.jsonl"
        temporal_packets_path = output / "temporal_review_packets.jsonl"
        private_map_path = output / "private_packet_map.json"
        packet_schema_path = output / "review_packet.schema.json"
        _write_jsonl(construct_packets_path, construct_packets)
        _write_jsonl(temporal_packets_path, temporal_packets)
        _write_json(private_map_path, private_map)
        _write_json(packet_schema_path, REVIEW_PACKET_SCHEMA)

        render_summary = json.loads(render_path.read_text(encoding="utf-8"))
        expected_prompt_rows = render_summary.get("rendered_prompt_count")
        if expected_prompt_rows is None:
            expected_prompt_rows = render_summary.get("prompt_count")
        if expected_prompt_rows is None and isinstance(render_summary.get("counts"), dict):
            expected_prompt_rows = render_summary["counts"].get("rendered_prompts")
        prompt_coverage_matches = expected_prompt_rows is None or expected_prompt_rows == temporal_report["counts"]["prompt_rows"]
        if not prompt_coverage_matches:
            raise ValueError(
                f"Rendered prompt count does not match summary: {temporal_report['counts']['prompt_rows']} != {expected_prompt_rows}"
            )

        lineage_validation = None
        if stage2_path is not None:
            stage2_sha256 = sha256_file(stage2_path)
            if lineage_manifest_path is not None:
                world_state_sha256 = sha256_file(world_path)
                bound = verify_bound_lineage_manifest(
                    lineage_manifest_path,
                    stage2_path=stage2_path,
                    stage3_path=world_path,
                    stage4_path=stage4_path,
                    stage2_sha256=stage2_sha256,
                    stage3_sha256=world_state_sha256,
                    stage4_sha256=stage4_digest.hexdigest(),
                )
                lineage_validation = bound.get("identity")
                if isinstance(lineage_validation, dict):
                    lineage_validation = {
                        **lineage_validation,
                        "identity_passed": lineage_validation.get("passed"),
                        "passed": bound["passed"],
                        "bound_manifest": bound,
                    }
                else:
                    lineage_validation = {"passed": False, "bound_manifest": bound}
            else:
                lineage_validation = _validate_stage234(Path(stage2_path), world_path, stage4_path)

        deterministic_summary = {
            "report_type": "automated_consistency_audit",
            "report_version": AUDIT_VERSION,
            "created_at_utc": _utc_now(),
            "coverage": {
                "stage4_rows": rows_total,
                "stage4_unique_case_ids": len(seen_ids),
                "world_state_entries": world_lookup.count(),
                "stage2_reconstruction": "available" if stage2_path is not None else "unavailable",
                "stage2_content_validated": lineage_validation is not None and lineage_validation["passed"],
                "stage2_limitation": None
                if stage2_path is not None
                else "Stage 2 is absent; the audit cannot independently reconstruct raw historical repair extraction.",
                "rendered_prompts": temporal_report["counts"]["prompt_rows"],
                "render_summary_count_matches": prompt_coverage_matches,
            },
            "status_counts": dict(sorted(status_counts.items())),
            "finding_counts": dict(sorted(finding_counts.items())),
            "track_counts": dict(sorted(track_counts.items())),
            "class_counts": dict(sorted(class_counts.items())),
            "subtype_counts": dict(sorted(subtype_counts.items())),
            "missing_id_rows": missing_id_rows,
            "automated_temporal_gate_passed": temporal_report["passed_automated_gate"],
            "lineage_validation": lineage_validation,
            "note": "Unsupported checks are coverage gaps, not agreement. Codex review is error discovery, not ground truth.",
        }
        summary_path = output / "deterministic_summary.json"
        _write_json(summary_path, deterministic_summary)

        artifacts = {}
        for name, path in {
            "deterministic_summary": summary_path,
            "deterministic_findings": findings_path,
            "deterministic_case_status": statuses_path,
            "temporal_audit": output / "temporal_audit.json",
            "construct_review_packets": construct_packets_path,
            "temporal_review_packets": temporal_packets_path,
            "private_packet_map": private_map_path,
            "review_packet_schema": packet_schema_path,
        }.items():
            artifacts[name] = {"path": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}

        manifest = {
            "manifest_type": "automated_consistency_audit",
            "manifest_version": AUDIT_VERSION,
            "created_at_utc": _utc_now(),
            "output_dir": str(output.resolve()),
            "inputs": {
                "classified_benchmark": str(stage4_path.resolve()),
                "classified_benchmark_sha256": stage4_digest.hexdigest(),
                "world_state": str(world_path.resolve()),
                "world_state_sha256": world_state_sha256 or sha256_file(world_path),
                "stage4_schema": str(schema_path.resolve()),
                "stage4_schema_sha256": sha256_file(schema_path),
                "construct_sample": str(construct_path.resolve()),
                "construct_sample_sha256": sha256_file(construct_path),
                "rendered_prompts": str(prompts_path.resolve()),
                "rendered_prompts_sha256": sha256_file(prompts_path),
                "render_summary": str(render_path.resolve()),
                "render_summary_sha256": sha256_file(render_path),
                "stage2": str(Path(stage2_path).resolve()) if stage2_path is not None else None,
                "stage2_sha256": stage2_sha256,
                "lineage_manifest": str(Path(lineage_manifest_path).resolve()) if lineage_manifest_path else None,
                "lineage_manifest_sha256": sha256_file(lineage_manifest_path) if lineage_manifest_path else None,
            },
            "parameters": {
                "construct_review_size": construct_review_size,
                "temporal_review_size": temporal_review_size,
                "seed": seed,
            },
            "git": starting_git_state,
            "world_state_index": {"source": world_lookup.index_source, "path": str(world_lookup.index_path)},
            "counts": {
                "stage4_rows": rows_total,
                "construct_review_packets": len(construct_packets),
                "temporal_review_packets": len(temporal_packets),
            },
            "validation": {
                "passed": (
                    (lineage_validation is None or lineage_validation["passed"])
                    and temporal_report["passed_automated_gate"]
                    and prompt_coverage_matches
                    and not any(code in finding_counts for code in INTEGRITY_CODES)
                ),
                "lineage_passed": lineage_validation is None or lineage_validation["passed"],
                "temporal_gate_passed": temporal_report["passed_automated_gate"],
                "render_coverage_passed": prompt_coverage_matches,
                "integrity_gate_passed": not any(code in finding_counts for code in INTEGRITY_CODES),
            },
            "artifacts": artifacts,
        }
        manifest_schema_path = Path(__file__).resolve().parents[1] / "schemas" / "automated_consistency_audit.schema.json"
        manifest_schema = json.loads(manifest_schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(manifest_schema)
        manifest_errors = list(Draft202012Validator(manifest_schema).iter_errors(manifest))
        if manifest_errors:
            raise ValueError(f"Generated audit manifest is invalid: {manifest_errors[0].message}")
        manifest_path = output / "manifest.json"
        _write_json(manifest_path, manifest)
        return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exhaustive consistency checks and Codex-assisted error discovery."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Scan Stage 4/3 and export blinded review packets.")
    run.add_argument("--classified-benchmark", required=True)
    run.add_argument("--world-state", required=True)
    run.add_argument("--stage4-schema", required=True)
    run.add_argument("--construct-sample", required=True)
    run.add_argument("--rendered-prompts", required=True)
    run.add_argument("--render-summary", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--construct-review-size", type=int, default=450)
    run.add_argument("--temporal-review-size", type=int, default=50)
    run.add_argument("--seed", type=int, default=13)
    run.add_argument("--stage2")
    run.add_argument("--lineage-manifest", help="Optional hash-bound v2 lineage result to reuse.")
    run.add_argument("--cache-dir", default=".cache/automated_audit")

    review = subparsers.add_parser("review-codex", help="Review blinded packet shards with Codex.")
    review.add_argument("--manifest", required=True)
    review.add_argument("--model", default="gpt-5.6-sol")
    review.add_argument("--shard-size", type=int, default=10)
    review.add_argument("--workers", type=int, default=3)
    review.add_argument("--retries", type=int, default=2)

    finalize = subparsers.add_parser("finalize", help="Merge AI findings into conservative dispositions.")
    finalize.add_argument("--manifest", required=True)
    finalize.add_argument("--reviews", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "run":
        manifest = run_audit(
            classified_benchmark_path=args.classified_benchmark,
            world_state_path=args.world_state,
            stage4_schema_path=args.stage4_schema,
            construct_sample_path=args.construct_sample,
            rendered_prompts_path=args.rendered_prompts,
            render_summary_path=args.render_summary,
            output_dir=args.output_dir,
            construct_review_size=args.construct_review_size,
            temporal_review_size=args.temporal_review_size,
            seed=args.seed,
            stage2_path=args.stage2,
            lineage_manifest_path=args.lineage_manifest,
            cache_dir=args.cache_dir,
        )
        print(json.dumps(manifest["counts"], sort_keys=True))
        return 0 if manifest["validation"]["passed"] else 1
    from automated_audit_codex import finalize_audit, run_codex_reviews

    if args.command == "review-codex":
        report = run_codex_reviews(
            manifest_path=args.manifest,
            model=args.model,
            shard_size=args.shard_size,
            workers=args.workers,
            retries=args.retries,
        )
    else:
        report = finalize_audit(manifest_path=args.manifest, reviews_path=args.reviews)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
