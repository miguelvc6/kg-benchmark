import hashlib
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import ijson
import requests

from . import config


def is_qid(value):
    """Return True if the value looks like a Wikidata item id (Q*)."""
    if not isinstance(value, str):
        return False
    return bool(config.QID_EXACT_PATTERN.fullmatch(value.strip()))


def is_pid(value):
    """Return True if the value looks like a Wikidata property id (P*)."""
    if not isinstance(value, str):
        return False
    return bool(config.PID_EXACT_PATTERN.fullmatch(value.strip()))


def is_entity_or_property_id(value):
    """Return True for valid QIDs or PIDs."""
    return is_qid(value) or is_pid(value)


def utc_now_iso():
    """Return a UTC timestamp string in ISO 8601 format (second precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_text(text):
    """Lowercase and collapse whitespace while keeping digits/letters/punctuation."""
    if not isinstance(text, str):
        return ""
    out = text.lower()
    out = re.sub(r"[^\w\s\-:/\.]", " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def read_json(path):
    """Read JSON from disk and return the decoded payload."""
    with open(Path(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj):
    """JSON serializer fallback for Decimal and Path."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def iter_jsonl(path):
    """Yield objects from a JSONL file."""
    with open(Path(path), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_repairs(path):
    """
    Supports:
      - .jsonl: one object per line
      - .json: either a JSON array, or a single JSON object (not expected here)
    For large arrays, uses ijson if available.
    """
    path = Path(path)
    if path.suffix == ".jsonl":
        yield from iter_jsonl(path)
        return

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

    obj = read_json(path)
    if isinstance(obj, dict):
        yield obj
    elif isinstance(obj, list):
        for entry in obj:
            if isinstance(entry, dict):
                yield entry
    else:
        raise ValueError(f"Unsupported JSON content in {path}: {type(obj)}")


def count_repairs(path):
    """Return count of repairs in a JSON or JSONL file (best-effort)."""
    path = Path(path)
    if path.suffix == ".jsonl":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return sum(1 for _ in fh)
        except Exception:
            return None
    if ijson is not None:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return sum(1 for _ in ijson.items(fh, "item"))
        except Exception:
            return None
    return None


def safe_get(payload, *keys, default=None):
    """Traverse nested dicts safely and return default on missing keys."""
    cur = payload
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_cached_repairs(path):
    """Return previously generated repair dataset if available."""
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            cached = json.load(fh)
    except Exception as exc:
        print(f"[!] Failed to read existing repairs file ({file_path}): {exc}")
        return None
    if isinstance(cached, list) and cached:
        print(f"[+] {file_path} exists. Loaded {len(cached)} repairs from disk. Skipping rebuild.")
        return cached
    print(f"[!] {file_path} is empty or malformed. Recomputing repairs...")
    return None


def load_jsonl_ids(jsonl_path):
    """Return (line_count, set(ids)) from a JSONL file, skipping malformed lines."""
    file_path = Path(jsonl_path)
    if not file_path.exists():
        return 0, set()
    seen_ids = set()
    line_count = 0
    with open(file_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            line_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                entry_id = record.get("id")
                if entry_id:
                    seen_ids.add(entry_id)
    return line_count, seen_ids


def build_candidate_key(item):
    """Return a stable resume key for a candidate based on report provenance."""
    parts = [
        item.get("qid") or "",
        item.get("property_id") or "",
        item.get("fix_date") or "",
        item.get("report_revision_old") or "",
        item.get("report_revision_new") or "",
    ]
    return "|".join(str(part) for part in parts)


def build_coarse_key(qid, pid, violation_type):
    """Return a coarse resume key when candidate provenance is unavailable."""
    return f"{qid}|{pid}|{violation_type or ''}"


def load_resume_stats(stats_path, include_coarse=False):
    """Load resume metadata from a stats JSONL file."""
    file_path = Path(stats_path) if stats_path else None
    if not file_path or not file_path.exists():
        return {
            "line_count": 0,
            "processed_keys": set(),
            "coarse_keys": set(),
            "last_record": None,
            "has_candidate_keys": False,
        }
    line_count = 0
    processed_keys = set()
    coarse_keys = set()
    last_record = None
    has_candidate_keys = False
    with open(file_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            line_count += 1
            last_record = record
            candidate_key = record.get("candidate_key")
            if candidate_key:
                processed_keys.add(candidate_key)
                has_candidate_keys = True
            if include_coarse:
                qid = record.get("qid")
                pid = record.get("property")
                violation_type = record.get("violation_type")
                if qid and pid:
                    coarse_keys.add(build_coarse_key(qid, pid, violation_type))
    return {
        "line_count": line_count,
        "processed_keys": processed_keys,
        "coarse_keys": coarse_keys,
        "last_record": last_record,
        "has_candidate_keys": has_candidate_keys,
    }


def load_resume_checkpoint(checkpoint_path):
    """Load a resume checkpoint from disk."""
    if not checkpoint_path:
        return None
    file_path = Path(checkpoint_path)
    if not file_path.exists():
        print(f"[!] Resume checkpoint not found: {file_path}")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        print(f"[!] Failed to read resume checkpoint ({file_path}): {exc}")
        return None
    if not isinstance(payload, dict):
        print(f"[!] Resume checkpoint malformed: {file_path}")
        return None
    return payload


def write_resume_checkpoint(checkpoint_path, payload):
    """Persist resume checkpoint atomically."""
    if not checkpoint_path:
        return
    path = Path(checkpoint_path)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(temp_path, path)


def append_jsonl_record(file_handle, record):
    """Append a single JSONL record to an open file handle."""
    file_handle.write(json.dumps(record, ensure_ascii=True))
    file_handle.write("\n")


def compile_jsonl_to_json(jsonl_path, json_path):
    """Compile JSONL to a JSON array via a temp file and atomic rename."""
    jsonl_path = Path(jsonl_path)
    json_path = Path(json_path)
    temp_path = json_path.with_suffix(json_path.suffix + ".tmp")
    with open(jsonl_path, "r", encoding="utf-8") as src, open(temp_path, "w", encoding="utf-8") as dst:
        dst.write("[")
        first = True
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not first:
                dst.write(",\n")
            else:
                first = False
            dst.write(json.dumps(record, ensure_ascii=True))
        dst.write("]\n")
    os.replace(temp_path, json_path)


def pick_label(entity, lang="en"):
    """Return preferred label for an entity, falling back to any language."""
    if not entity:
        return None
    labels = entity.get("labels", {})
    if lang in labels:
        return labels[lang].get("value")
    if labels:
        first = next(iter(labels.values()))
        return first.get("value")
    return None


def pick_description(entity, lang="en"):
    """Return preferred description for an entity, falling back to any language."""
    if not entity:
        return None
    descriptions = entity.get("descriptions", {})
    if lang in descriptions:
        return descriptions[lang].get("value")
    if descriptions:
        first = next(iter(descriptions.values()))
        return first.get("value")
    return None


def chunked(iterable, size):
    """Yield iterable slices of fixed size (used for batched API lookups)."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def parse_iso8601(raw_ts):
    """Parse API timestamps while stripping milliseconds and enforcing UTC."""
    if not raw_ts:
        return None
    normalized = raw_ts[:-1] + "+00:00" if raw_ts.endswith("Z") else raw_ts
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.replace(microsecond=0)


def format_timestamp(dt):
    """Return a MediaWiki-compatible string (UTC, second precision)."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_revision_window(report_date):
    """Derive the [start, end] revision window relative to the report fix date."""
    end_dt = parse_iso8601(report_date)
    if not end_dt:
        return None, None
    start_dt = end_dt - timedelta(days=config.REVISION_LOOKBACK_DAYS)
    return format_timestamp(start_dt), format_timestamp(end_dt)


def format_datavalue(value):
    """Human readable rendering of Wikibase datavalue payloads."""
    if isinstance(value, dict):
        if "time" in value:
            return str(value["time"])
        if "id" in value:
            return str(value["id"])
        if "text" in value:
            lang = value.get("language")
            return f"{value['text']}@{lang}" if lang else value["text"]
        if "amount" in value:
            unit = value.get("unit", "")
            return f"{value['amount']} {unit}".strip()
        if "latitude" in value and "longitude" in value:
            lat = str(value["latitude"])
            lon = str(value["longitude"])
            return f"{lat},{lon}"
    return str(value)


def summarize_claims(claims):
    """Convert a list of statements into a hashable signature and display values."""
    if not claims:
        return ("MISSING",), ["MISSING"]
    signature_parts = []
    display_values = []
    for claim in claims:
        snak = claim.get("mainsnak", {})
        snak_type = snak.get("snaktype", "").upper() or "UNKNOWN"
        if snak_type == "VALUE":
            value_str = format_datavalue(snak.get("datavalue", {}).get("value"))
        else:
            value_str = snak_type
        signature_parts.append(f"{snak_type}:{value_str}")
        display_values.append(value_str)
    if not signature_parts:
        return ("MISSING",), ["MISSING"]
    return tuple(sorted(signature_parts)), display_values


def _normalize_constraint_claim(claim):
    """Canonicalize a P2302 statement for deterministic hashing."""
    mainsnak = claim.get("mainsnak", {})
    snak_type = mainsnak.get("snaktype", "").upper()
    datavalue = mainsnak.get("datavalue", {})
    raw_value = datavalue.get("value")
    constraint_qid = None
    if isinstance(raw_value, dict):
        constraint_qid = raw_value.get("id") or raw_value.get("text")
    elif raw_value is not None:
        constraint_qid = format_datavalue(raw_value)
    qualifiers = []
    qualifier_map = claim.get("qualifiers", {})
    for qualifier_pid in sorted(qualifier_map.keys()):
        rendered_values = []
        for qualifier in qualifier_map.get(qualifier_pid, []):
            q_datavalue = qualifier.get("datavalue")
            if not q_datavalue:
                continue
            q_raw = q_datavalue.get("value")
            rendered_values.append(format_datavalue(q_raw))
        rendered_values.sort()
        qualifiers.append(
            {
                "property_id": qualifier_pid,
                "values": rendered_values,
            }
        )
    return {
        "constraint_qid": constraint_qid,
        "snaktype": snak_type,
        "rank": claim.get("rank"),
        "qualifiers": qualifiers,
    }


def canonicalize_json_structure(payload):
    """Return the canonical serialization used for hashing constraint signatures."""
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def signature_p2302(claims):
    """
    Return a deterministic signature for constraint statements.
    The signature ignores ordering by sorting constraint types and qualifier payloads before hashing.
    """
    normalized = [_normalize_constraint_claim(claim) for claim in claims or []]
    normalized.sort(
        key=lambda entry: (
            entry.get("constraint_qid") or "",
            entry.get("snaktype") or "",
            entry.get("rank") or "",
            json.dumps(entry.get("qualifiers", []), sort_keys=True, ensure_ascii=True),
        )
    )
    serialized = canonicalize_json_structure(normalized)
    digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()
    constraint_types = sorted({entry.get("constraint_qid") for entry in normalized if entry.get("constraint_qid")})
    return {
        "normalized": normalized,
        "signature": serialized,
        "hash": digest,
        "constraint_types": constraint_types,
    }


def _build_constraint_delta(previous_signature, current_signature, include_snapshots):
    """Assemble before/after hashes (optionally include raw constraints)."""
    delta = {
        "signature_before_raw": previous_signature["signature"],
        "signature_after_raw": current_signature["signature"],
        "signature_before": previous_signature["normalized"],
        "signature_after": current_signature["normalized"],
        "hash_before": previous_signature["hash"],
        "hash_after": current_signature["hash"],
        "changed_constraint_types": sorted(
            set(previous_signature.get("constraint_types") or [])
            ^ set(current_signature.get("constraint_types") or [])
        ),
    }
    if include_snapshots:
        delta["old_constraints"] = previous_signature["normalized"]
        delta["new_constraints"] = current_signature["normalized"]
    return delta


def classify_action(previous_signature, current_signature):
    """Infer CRUD action type between two claim signatures."""
    if current_signature == ("MISSING",):
        return "DELETE"
    if previous_signature == ("MISSING",):
        return "CREATE"
    return "UPDATE"


def get_json(params=None, *, endpoint=config.API_ENDPOINT, with_format=True):
    """Wrapper around requests.get with retries and default MediaWiki params."""
    query = dict(params or {})
    if with_format:
        query.setdefault("format", "json")
        query.setdefault("formatversion", 2)
    for attempt in range(4):
        try:
            response = requests.get(
                endpoint,
                headers=config.HEADERS,
                params=query if query else None,
                timeout=config.API_TIMEOUT,
            )
            if response.status_code == 200:
                return response.json()
            if response.status_code == 429:
                sleep_for = 2**attempt
                print(f"    [!] Rate limited. Sleeping {sleep_for}s...")
                time.sleep(sleep_for)
            else:
                print(f"    [!] HTTP {response.status_code} for {endpoint}")
        except Exception as exc:
            print(f"    [!] Exception: {exc}")
        time.sleep(0.1)
    return None


def parse_report_violation_type_qids(raw_text):
    """Extract normalized QIDs from the raw violation type string."""
    if not raw_text or not isinstance(raw_text, str):
        return []
    seen = set()
    ordered = []
    for match in config.REPORT_VIOLATION_QID_PATTERN.finditer(raw_text):
        qid = f"Q{match.group(1)}"
        if qid in seen:
            continue
        seen.add(qid)
        ordered.append(qid)
    return ordered


def extract_qids_from_sequence(values):
    """Return ordered list of QIDs detected inside a list-like payload."""
    if not values:
        return []
    iterable = values if isinstance(values, list) else [values]
    qids = []
    for value in iterable:
        if is_qid(value):
            qids.append(value)
    return qids


def _null_label_entry():
    return {
        "label_en": None,
        "description_en": None,
    }


def add_resolved_list_fields(container, field_name, ids, resolved_lookup):
    """Attach *_labels_en/_descriptions_en based on an id list."""
    if not container or not ids:
        return
    labels = []
    descriptions = []
    for entity_id in ids:
        resolution = resolved_lookup.get(entity_id) or _null_label_entry()
        labels.append(resolution["label_en"])
        descriptions.append(resolution["description_en"])
    container[f"{field_name}_labels_en"] = labels
    container[f"{field_name}_descriptions_en"] = descriptions


def ensure_signature_structures(constraint_delta):
    """Backfill structured signatures and *_raw fields for older records."""
    if not constraint_delta or not isinstance(constraint_delta, dict):
        return
    for key in ("signature_before", "signature_after"):
        raw_key = f"{key}_raw"
        payload = constraint_delta.get(key)
        if isinstance(payload, str):
            if raw_key not in constraint_delta:
                constraint_delta[raw_key] = payload
            try:
                constraint_delta[key] = json.loads(payload)
            except json.JSONDecodeError:
                constraint_delta[key] = []
        elif payload is None:
            constraint_delta[key] = []
        elif isinstance(payload, list):
            if raw_key not in constraint_delta and isinstance(constraint_delta.get(raw_key), str):
                continue
        else:
            constraint_delta[key] = []


def collect_constraint_related_ids(constraint_delta):
    """Gather every QID/PID referenced inside the constraint signatures."""
    ids = set()
    if not constraint_delta or not isinstance(constraint_delta, dict):
        return ids
    ensure_signature_structures(constraint_delta)
    for field in ("signature_before", "signature_after"):
        signature = constraint_delta.get(field) or []
        for constraint in signature:
            if not isinstance(constraint, dict):
                continue
            constraint_qid = constraint.get("constraint_qid")
            if is_qid(constraint_qid):
                ids.add(constraint_qid)
            qualifiers = constraint.get("qualifiers") or []
            for qualifier in qualifiers:
                qualifier_pid = qualifier.get("property_id")
                if is_pid(qualifier_pid):
                    ids.add(qualifier_pid)
                for value in qualifier.get("values") or []:
                    if is_qid(value) or is_pid(value):
                        ids.add(value)
    return ids


def build_readable_constraints(signature_entries, resolved_lookup):
    """Return (constraints_readable_en, rule_summaries_en) for a signature list."""
    readable = []
    summaries = []
    entries = signature_entries or []
    for constraint in entries:
        if not isinstance(constraint, dict):
            continue
        constraint_qid = constraint.get("constraint_qid")
        constraint_resolution = resolved_lookup.get(constraint_qid) or _null_label_entry()
        readable_entry = {
            "constraint_type": {
                "id": constraint_qid,
                "label_en": constraint_resolution["label_en"],
                "description_en": constraint_resolution["description_en"],
            },
            "rank": constraint.get("rank"),
            "snaktype": constraint.get("snaktype"),
            "parameters": {},
        }
        summary_segments = []
        qualifiers = constraint.get("qualifiers") or []
        for qualifier in qualifiers:
            qualifier_pid = qualifier.get("property_id")
            property_resolution = resolved_lookup.get(qualifier_pid) or _null_label_entry()
            parameter_key = qualifier_pid
            parameter_values = []
            summary_values = []
            for raw_value in qualifier.get("values") or []:
                if is_qid(raw_value) or is_pid(raw_value):
                    value_resolution = resolved_lookup.get(raw_value) or _null_label_entry()
                    parameter_values.append(
                        {
                            "id": raw_value,
                            "label_en": value_resolution["label_en"],
                            "description_en": value_resolution["description_en"],
                        }
                    )
                    summary_values.append(value_resolution["label_en"] or raw_value)
                else:
                    parameter_values.append({"value": raw_value})
                    summary_values.append(raw_value)
            readable_entry["parameters"].setdefault(parameter_key, []).extend(parameter_values)
            prop_label = property_resolution["label_en"] or qualifier_pid
            summary_body = ", ".join(summary_values) if summary_values else "unspecified"
            summary_segments.append(f"{prop_label}: {summary_body}")
        readable.append(readable_entry)
        constraint_label = readable_entry["constraint_type"]["label_en"] or constraint_qid or "Constraint"
        if summary_segments:
            summaries.append(f"{constraint_label}: " + "; ".join(summary_segments))
        else:
            summaries.append(f"{constraint_label}: no qualifiers recorded")
    return readable, summaries


def annotate_constraint_delta(constraint_delta, resolved_lookup):
    """Decorate constraint deltas with readable mirrors and ensure *_raw fields."""
    if not constraint_delta or not isinstance(constraint_delta, dict):
        return
    ensure_signature_structures(constraint_delta)
    before_readable, before_summaries = build_readable_constraints(
        constraint_delta.get("signature_before"), resolved_lookup
    )
    after_readable, after_summaries = build_readable_constraints(
        constraint_delta.get("signature_after"), resolved_lookup
    )
    constraint_delta["constraints_readable_en"] = {
        "before": before_readable,
        "after": after_readable,
    }
    constraint_delta["rule_summaries_en"] = {
        "before": before_summaries,
        "after": after_summaries,
    }


def enrich_repair_entry(entry, resolver):
    """Add human-readable mirrors for ids while preserving machine-stable fields."""
    if not isinstance(entry, dict) or resolver is None:
        return entry
    violation_context = entry.setdefault("violation_context", {})
    repair_target = entry.setdefault("repair_target", {})
    persistence_check = entry.setdefault("persistence_check", {})

    ids_to_resolve = set()

    qid = entry.get("qid")
    if is_qid(qid):
        ids_to_resolve.add(qid)
    property_id = entry.get("property")
    if is_pid(property_id):
        ids_to_resolve.add(property_id)

    report_type_raw = violation_context.get("report_violation_type")
    if report_type_raw and "report_violation_type_raw" not in violation_context:
        violation_context["report_violation_type_raw"] = report_type_raw
    report_qids = parse_report_violation_type_qids(report_type_raw)
    ids_to_resolve.update(report_qids)

    violation_value_qids = extract_qids_from_sequence(violation_context.get("value"))
    violation_value_current_qids = extract_qids_from_sequence(violation_context.get("value_current_2026"))
    ids_to_resolve.update(violation_value_qids)
    ids_to_resolve.update(violation_value_current_qids)

    persistence_qids = extract_qids_from_sequence(persistence_check.get("current_value_2026"))
    ids_to_resolve.update(persistence_qids)

    old_value_qids = extract_qids_from_sequence(repair_target.get("old_value"))
    new_value_qids = extract_qids_from_sequence(repair_target.get("new_value"))
    repair_value_qids = extract_qids_from_sequence(repair_target.get("value"))
    ids_to_resolve.update(old_value_qids)
    ids_to_resolve.update(new_value_qids)
    ids_to_resolve.update(repair_value_qids)

    constraint_delta = repair_target.get("constraint_delta")
    ids_to_resolve.update(collect_constraint_related_ids(constraint_delta))

    resolved_lookup = resolver.resolve(sorted(ids_to_resolve))

    def resolved_info(entity_id):
        if not entity_id:
            return _null_label_entry()
        return resolved_lookup.get(entity_id) or _null_label_entry()

    qid_resolution = resolved_info(qid)
    entry["qid_label_en"] = qid_resolution["label_en"]
    entry["qid_description_en"] = qid_resolution["description_en"]

    property_resolution = resolved_info(property_id)
    entry["property_label_en"] = property_resolution["label_en"]
    entry["property_description_en"] = property_resolution["description_en"]

    violation_context["report_violation_type_qids"] = report_qids
    if report_qids:
        add_resolved_list_fields(violation_context, "report_violation_type", report_qids, resolved_lookup)

    add_resolved_list_fields(violation_context, "value", violation_value_qids, resolved_lookup)
    add_resolved_list_fields(violation_context, "value_current_2026", violation_value_current_qids, resolved_lookup)
    add_resolved_list_fields(persistence_check, "current_value_2026", persistence_qids, resolved_lookup)
    add_resolved_list_fields(repair_target, "old_value", old_value_qids, resolved_lookup)
    add_resolved_list_fields(repair_target, "new_value", new_value_qids, resolved_lookup)
    add_resolved_list_fields(repair_target, "value", repair_value_qids, resolved_lookup)

    if constraint_delta:
        annotate_constraint_delta(constraint_delta, resolved_lookup)

    return entry


def enrich_repair_entries(entries, resolver):
    """Enrich an entire Stage-2 dataset in-place."""
    if not entries or resolver is None:
        return entries
    for entry in entries:
        enrich_repair_entry(entry, resolver)
    return entries
