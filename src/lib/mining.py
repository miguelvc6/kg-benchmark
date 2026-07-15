import json
import os
import random
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import mwclient
from tqdm import tqdm

from . import config

# Lazy MediaWiki site handle (initialized on first use)
SITE = None
_LAST_REPORT_REQUEST_MONOTONIC = None
_CANDIDATE_CHECKPOINT_VERSION = 1


def _write_json_atomic(path, payload, *, indent=2):
    """Write JSON without exposing a partially written artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=indent, ensure_ascii=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _candidate_checkpoint_paths(path):
    path = Path(path)
    return (
        path.with_name(f"{path.stem}.checkpoint.json"),
        path.with_name(f"{path.stem}.partial.jsonl"),
    )


def _wait_for_report_request_slot():
    """Enforce a process-wide minimum interval between report API requests."""
    global _LAST_REPORT_REQUEST_MONOTONIC
    interval = max(0.0, float(config.REPORT_REQUEST_INTERVAL_SECONDS))
    now = time.monotonic()
    if _LAST_REPORT_REQUEST_MONOTONIC is not None:
        remaining = interval - (now - _LAST_REPORT_REQUEST_MONOTONIC)
        if remaining > 0:
            time.sleep(remaining)
            now = time.monotonic()
    _LAST_REPORT_REQUEST_MONOTONIC = now


def _retry_after_seconds(exc):
    """Return the delay requested by an HTTP Retry-After header, if present."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("Retry-After") or headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(value))
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _report_retry_delay(exc, attempt):
    requested = _retry_after_seconds(exc)
    if requested is None:
        requested = min(
            float(config.REPORT_RETRY_BASE_SECONDS) * (2**attempt),
            float(config.REPORT_RETRY_MAX_SECONDS),
        )
    jitter_limit = max(0.0, float(config.REPORT_RETRY_JITTER_SECONDS))
    return requested + random.uniform(0.0, jitter_limit)


def is_valid_violation_section(section):
    """Return True if a report section represents a real violation bucket."""
    if not section:
        return False
    if section in config.INVALID_REPORT_SECTIONS:
        return False
    if section.strip().lower() in {"unknown"}:
        return False
    return True


def normalize_report_violation_type(section):
    """Strip lightweight wiki templates/markup for a cleaner display-only header."""
    if not section:
        return None
    normalized = re.sub(r"\{\{[^}]+\}\}", "", section).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized or None


def get_wikidata_site():
    """Lazy initializer for the MediaWiki client used during candidate mining."""
    if mwclient is None:
        raise RuntimeError(
            "mwclient is required to build 01_repair_candidates.json. Install it via 'pip install mwclient'."
        )
    global SITE
    if SITE is None:
        attempts = max(1, int(config.REPORT_FETCH_ATTEMPTS))
        last_error = None
        for attempt in range(attempts):
            try:
                _wait_for_report_request_slot()
                SITE = mwclient.Site("www.wikidata.org", clients_useragent=config.HEADERS["User-Agent"])
                break
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    break
                delay = _report_retry_delay(exc, attempt)
                print(
                    f"    [!] Wikidata client initialization attempt {attempt + 1}/{attempts} failed: {exc}. "
                    f"Retrying in {delay:.1f}s."
                )
                time.sleep(delay)
        if SITE is None:
            raise RuntimeError(f"Failed to initialize the Wikidata client: {last_error}")
    return SITE


def get_report_page_title(property_id):
    """Return the standard constraint report page title for a property."""
    return f"Wikidata:Database reports/Constraint violations/{property_id}"


def fetch_all_active_properties():
    """Return all properties listed on the Constraint violations summary page."""
    try:
        site = get_wikidata_site()
    except RuntimeError as exc:
        raise RuntimeError(f"Cannot auto-discover properties: {exc}") from exc
    summary_page = None
    text = None
    last_error = None
    attempts = max(1, int(config.REPORT_FETCH_ATTEMPTS))
    for attempt in range(attempts):
        try:
            if summary_page is None:
                _wait_for_report_request_slot()
                summary_page = site.pages["Wikidata:Database reports/Constraint violations/Summary"]
                if not summary_page.exists:
                    raise RuntimeError("Constraint-violation summary page was not found.")
            _wait_for_report_request_slot()
            text = summary_page.text()
            break
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                break
            delay = _report_retry_delay(exc, attempt)
            print(
                f"    [!] Summary fetch attempt {attempt + 1}/{attempts} failed: {exc}. "
                f"Retrying in {delay:.1f}s."
            )
            time.sleep(delay)
    if text is None:
        raise RuntimeError(f"Failed to read constraint-violation summary page: {last_error}")
    found_props = sorted(set(re.findall(r"P\d+", text)))
    print(f"[*] Auto-discovered {len(found_props)} properties with active reports.")
    return found_props


def extract_qids_with_context(text):
    """
    Parses report page text to associate QIDs with their constraint section.
    Returns: dict {qid: set(constraint_types)}
    """
    if not text:
        return {}

    qid_map = {}
    current_section = None

    # Matches headers like "== Format ==" or "=== Single value ==="
    header_pattern = re.compile(r"^={2,}\s*([^=]+?)\s*={2,}\s*$")
    cleaner = re.compile(r"['\"\[\]\{\}]| violations$| matches$", re.IGNORECASE)

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Detect section header
        header_match = header_pattern.match(line)
        if header_match:
            raw_header = header_match.group(1).strip()
            clean_header = cleaner.sub("", raw_header).strip()
            current_section = clean_header if is_valid_violation_section(clean_header) else None
            continue

        # Extract QIDs in this line
        qids = config.QID_PATTERN.findall(line)
        if not current_section:
            continue
        for qid in qids:
            if qid not in qid_map:
                qid_map[qid] = set()
            qid_map[qid].add(current_section)

    return qid_map


def mine_repairs(property_id, max_items=100):
    """Inspect report page history and return candidates with violation type context."""
    site = get_wikidata_site()
    print(f"[*] Mining history for {property_id}...")
    page = None
    revisions = None
    last_error = None
    attempts = max(1, int(config.REPORT_FETCH_ATTEMPTS))
    for attempt in range(attempts):
        try:
            if page is None:
                _wait_for_report_request_slot()
                page = site.pages[get_report_page_title(property_id)]
            _wait_for_report_request_slot()
            revisions = list(page.revisions(max_items=max_items, prop="content|timestamp|ids"))
            break
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                print(f"    [!] Report fetch attempt {attempt + 1}/{attempts} failed for {property_id}: {exc}")
                break
            delay = _report_retry_delay(exc, attempt)
            print(
                f"    [!] Report fetch attempt {attempt + 1}/{attempts} failed for {property_id}: {exc}. "
                f"Retrying in {delay:.1f}s."
            )
            time.sleep(delay)
    if revisions is None:
        raise RuntimeError(f"Failed to fetch report page for {property_id}: {last_error}")

    print(f"    Found {len(revisions)} revisions to analyze.")
    candidates = []
    revision_pairs = range(len(revisions) - 1)

    for i in tqdm(revision_pairs, desc=f"Diffing {property_id}", unit="pair", disable=not sys.stderr.isatty()):
        newer_rev = revisions[i]
        older_rev = revisions[i + 1]

        # Parse both revisions to get {QID: {Violations}}
        qids_old_map = extract_qids_with_context(older_rev.get("*", ""))
        qids_new_map = extract_qids_with_context(newer_rev.get("*", ""))

        # Find QIDs that disappeared from a specific section
        for qid, old_constraints in qids_old_map.items():
            new_constraints = qids_new_map.get(qid, set())
            fixed_constraints = old_constraints - new_constraints

            if fixed_constraints:
                if qid in qids_new_map:
                    # QID still present elsewhere (moved sections), so not a resolved violation.
                    continue
                timestamp_tuple = newer_rev.get("timestamp") or ()
                if len(timestamp_tuple) >= 6:
                    timestamp = datetime(*timestamp_tuple[:6]).isoformat()
                else:
                    timestamp = datetime.now(timezone.utc).isoformat()

                for c_type in fixed_constraints:
                    candidates.append(
                        {
                            "qid": qid,
                            "property_id": property_id,
                            "violation_type": c_type,  # <--- CAPTURED HERE
                            "fix_date": timestamp,
                            "report_revision_old": older_rev["revid"],
                            "report_revision_new": newer_rev["revid"],
                        }
                    )
    return candidates


def _new_candidate_checkpoint(target_properties, history_limit):
    return {
        "manifest_type": "repair_candidate_mining_checkpoint",
        "manifest_version": _CANDIDATE_CHECKPOINT_VERSION,
        "target_properties": list(target_properties),
        "history_limit": int(history_limit),
        "next_property_index": 0,
        "candidate_count": 0,
        "partial_bytes": 0,
    }


def _load_candidate_checkpoint(checkpoint_path, partial_path, history_limit):
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Stage 1 checkpoint is unreadable: {checkpoint_path}: {exc}") from exc
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("manifest_type") != "repair_candidate_mining_checkpoint"
        or checkpoint.get("manifest_version") != _CANDIDATE_CHECKPOINT_VERSION
    ):
        raise RuntimeError(f"Stage 1 checkpoint has an unsupported format: {checkpoint_path}")
    if checkpoint.get("history_limit") != int(history_limit):
        raise RuntimeError(
            f"Stage 1 checkpoint history limit {checkpoint.get('history_limit')!r} does not match "
            f"the requested limit {int(history_limit)}."
        )
    properties = checkpoint.get("target_properties")
    next_index = checkpoint.get("next_property_index")
    candidate_count = checkpoint.get("candidate_count")
    partial_bytes = checkpoint.get("partial_bytes")
    if (
        not isinstance(properties, list)
        or not properties
        or not all(isinstance(prop, str) and re.fullmatch(r"P\d+", prop) for prop in properties)
        or not isinstance(next_index, int)
        or not 0 <= next_index <= len(properties)
        or not isinstance(candidate_count, int)
        or candidate_count < 0
        or not isinstance(partial_bytes, int)
        or partial_bytes < 0
    ):
        raise RuntimeError(f"Stage 1 checkpoint is malformed: {checkpoint_path}")
    if not partial_path.is_file():
        raise RuntimeError(f"Stage 1 checkpoint is missing its partial candidate file: {partial_path}")
    actual_size = partial_path.stat().st_size
    if actual_size < partial_bytes:
        raise RuntimeError(
            f"Stage 1 partial candidate file is shorter than its checkpoint ({actual_size} < {partial_bytes} bytes)."
        )
    if actual_size != partial_bytes:
        with partial_path.open("r+b") as fh:
            fh.truncate(partial_bytes)
        print(f"[*] Discarded an uncheckpointed Stage 1 tail from {partial_path}.")

    candidates = []
    line_number = 0
    try:
        with partial_path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                if line.strip():
                    candidate = json.loads(line)
                    if not isinstance(candidate, dict):
                        raise ValueError("candidate record is not an object")
                    candidates.append(candidate)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(
            f"Stage 1 partial candidate file is invalid at or before line {line_number}: {partial_path}: {exc}"
        ) from exc
    if len(candidates) != candidate_count:
        raise RuntimeError(
            f"Stage 1 partial candidate count does not match its checkpoint "
            f"({len(candidates)} != {candidate_count})."
        )
    return checkpoint, candidates


def _append_candidate_checkpoint(partial_path, checkpoint_path, checkpoint, candidates):
    with partial_path.open("ab") as fh:
        for candidate in candidates:
            line = json.dumps(candidate, ensure_ascii=False, separators=(",", ":")) + "\n"
            fh.write(line.encode("utf-8"))
        fh.flush()
        os.fsync(fh.fileno())
        partial_bytes = fh.tell()
    checkpoint["next_property_index"] += 1
    checkpoint["candidate_count"] += len(candidates)
    checkpoint["partial_bytes"] = partial_bytes
    _write_json_atomic(checkpoint_path, checkpoint)


def ensure_repair_candidates_file(filename, history_limit=config.REPORT_HISTORY_DEPTH, *, force_refresh=False):
    """Load cached repair candidates or rebuild the file."""

    # Load from disk if available
    path = Path(filename)
    checkpoint_path, partial_path = _candidate_checkpoint_paths(path)
    if path.exists() and not force_refresh:
        with open(path, "r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if isinstance(cached, list) and cached:
            print(f"[+] {filename} exists. Loaded {len(cached)} candidates from disk.")
            return cached
        print(f"[!] {filename} is empty or malformed. Rebuilding...")

    checkpoint = None
    fresh_candidates = []
    if checkpoint_path.exists():
        checkpoint, fresh_candidates = _load_candidate_checkpoint(checkpoint_path, partial_path, history_limit)
        target_properties = checkpoint["target_properties"]
        print(
            f"[*] Resuming Stage 1 after {checkpoint['next_property_index']}/"
            f"{len(target_properties)} properties with {len(fresh_candidates)} checkpointed candidates."
        )
    else:
        if not config.TARGET_PROPERTIES:
            print("[*] No TARGET_PROPERTIES defined. Auto-discovering from summary page...")
            discovered = fetch_all_active_properties()
            if discovered:
                config.TARGET_PROPERTIES[:] = discovered
        if not config.TARGET_PROPERTIES:
            print("[!] Failed to identify any properties to mine.")
            return []
        target_properties = list(config.TARGET_PROPERTIES)
        path.parent.mkdir(parents=True, exist_ok=True)
        with partial_path.open("wb") as fh:
            fh.flush()
            os.fsync(fh.fileno())
        checkpoint = _new_candidate_checkpoint(target_properties, history_limit)
        _write_json_atomic(checkpoint_path, checkpoint)

    # Rebuild candidate list
    print(f"[!] {'Refreshing' if force_refresh else 'Missing'} {filename}. Mining fresh candidate list...")
    for prop in target_properties[checkpoint["next_property_index"] :]:
        property_candidates = mine_repairs(prop, max_items=history_limit)
        _append_candidate_checkpoint(partial_path, checkpoint_path, checkpoint, property_candidates)
        fresh_candidates.extend(property_candidates)

    # Save to disk
    _write_json_atomic(path, fresh_candidates)
    checkpoint_path.unlink(missing_ok=True)
    partial_path.unlink(missing_ok=True)
    print(f"[+] Done. Found {len(fresh_candidates)} candidates. Saved to {filename}.")
    return fresh_candidates


def deduplicate_candidates(candidates):
    """Deduplicate candidate list without dropping violation type information."""
    if not candidates:
        return [], {"duplicates_skipped": 0, "violation_type_merges": 0, "exact_duplicates": 0}
    exact_seen = set()
    base_seen = {}
    deduped = []
    duplicates_skipped = 0
    violation_type_merges = 0
    exact_duplicates = 0

    for item in candidates:
        qid = item.get("qid")
        pid = item.get("property_id")
        fix_date = item.get("fix_date")
        report_old = item.get("report_revision_old")
        report_new = item.get("report_revision_new")
        violation_type = item.get("violation_type")
        exact_key = (qid, pid, fix_date, report_old, report_new, violation_type)
        if exact_key in exact_seen:
            exact_duplicates += 1
            duplicates_skipped += 1
            continue
        exact_seen.add(exact_key)

        base_key = (qid, pid, fix_date, report_old, report_new)
        existing = base_seen.get(base_key)
        if existing:
            violation_type_merges += 1
            duplicates_skipped += 1
            if not existing.get("violation_type") and violation_type:
                existing["violation_type"] = violation_type
            merged_types = existing.setdefault("violation_types", [])
            if not merged_types and existing.get("violation_type"):
                merged_types.append(existing.get("violation_type"))
            if violation_type and violation_type not in merged_types:
                merged_types.append(violation_type)
            continue

        base_seen[base_key] = item
        deduped.append(item)

    return (
        deduped,
        {
            "duplicates_skipped": duplicates_skipped,
            "violation_type_merges": violation_type_merges,
            "exact_duplicates": exact_duplicates,
        },
    )


def build_report_provenance(candidate, property_id):
    """Return report metadata fields captured during Stage 1 mining."""
    provenance = {
        "report_fix_date": candidate.get("fix_date"),
        "report_revision_old": candidate.get("report_revision_old"),
        "report_revision_new": candidate.get("report_revision_new"),
    }
    page_title = candidate.get("report_page_title")
    if not page_title and property_id:
        page_title = get_report_page_title(property_id)
    if page_title:
        provenance["report_page_title"] = page_title
    return provenance
