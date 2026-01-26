import json
import re
from datetime import datetime, timezone
from pathlib import Path

import mwclient
from tqdm import tqdm

from . import config

# Lazy MediaWiki site handle (initialized on first use)
SITE = None


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
        SITE = mwclient.Site("www.wikidata.org", clients_useragent=config.HEADERS["User-Agent"])
    return SITE


def get_report_page_title(property_id):
    """Return the standard constraint report page title for a property."""
    return f"Wikidata:Database reports/Constraint violations/{property_id}"


def fetch_all_active_properties():
    """Return all properties listed on the Constraint violations summary page."""
    try:
        site = get_wikidata_site()
    except RuntimeError as exc:
        print(f"[!] Cannot auto-discover properties: {exc}")
        return []
    summary_page = site.pages["Wikidata:Database reports/Constraint violations/Summary"]
    if not summary_page.exists:
        print("[!] Summary page not found. Defaulting to empty property list.")
        return []
    try:
        text = summary_page.text()
    except Exception as exc:
        print(f"[!] Failed to read summary page: {exc}")
        return []
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
    try:
        page = site.pages[get_report_page_title(property_id)]
        revisions = list(page.revisions(max_items=max_items, prop="content|timestamp|ids"))
    except Exception as exc:
        print(f"    [!] Failed to fetch report page for {property_id}: {exc}")
        return []

    print(f"    Found {len(revisions)} revisions to analyze.")
    candidates = []
    revision_pairs = range(len(revisions) - 1)

    for i in tqdm(revision_pairs, desc=f"Diffing {property_id}", unit="pair"):
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


def ensure_repair_candidates_file(filename, history_limit=config.REPORT_HISTORY_DEPTH):
    """Load cached repair candidates or rebuild the file."""

    # Load from disk if available
    path = Path(filename)
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if isinstance(cached, list) and cached:
            print(f"[+] {filename} exists. Loaded {len(cached)} candidates from disk.")
            return cached
        print(f"[!] {filename} is empty or malformed. Rebuilding...")

    if not config.TARGET_PROPERTIES:
        print("[*] No TARGET_PROPERTIES defined. Auto-discovering from summary page...")
        discovered = fetch_all_active_properties()
        if discovered:
            config.TARGET_PROPERTIES[:] = discovered
    if not config.TARGET_PROPERTIES:
        print("[!] Failed to identify any properties to mine.")
        return []

    # Rebuild candidate list
    print(f"[!] {filename} missing. Mining fresh candidate list...")
    fresh_candidates = []
    for prop in config.TARGET_PROPERTIES:
        fresh_candidates.extend(mine_repairs(prop, max_items=history_limit))

    # Save to disk
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(fresh_candidates, fh, indent=2)
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
