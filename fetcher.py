import argparse
import gzip
import hashlib
import json
import re
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import ijson
import mwclient
import requests
from tqdm import tqdm

# User agent for polite API usage
HEADERS = {"User-Agent": "WikidataRepairEval/1.0 (PhD Research; mailto:miguel.vazquez@wu.ac.at)"}
# Base endpoints for Action API, REST history, and snapshot fetches
API_ENDPOINT = "https://www.wikidata.org/w/api.php"
REST_HISTORY_URL = "https://www.wikidata.org/w/rest.php/v1/page/{qid}/history"
ENTITY_DATA_URL = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"

# Fetch tuning knobs and data volume limits
STRICT_PERSISTENCE = True  # Drop candidates when current value missing
API_TIMEOUT = 30  # Seconds per HTTP request
REVISION_LOOKBACK_DAYS = 7  # Historical window size
MAX_HISTORY_PAGES = 8  # REST paging limit
MAX_PROPERTY_VALUES = 12  # Max values recorded per property
MAX_NEIGHBOR_EDGES = 50  # Max neighborhood edges captured
LATEST_DUMP_PATH = Path("data/latest-all.json.gz")  # 2025 dump location
WORLD_STATE_FILE = Path("data/03_world_state.json")  # Output for built contexts
# Limit processing to specific properties for debugging; leave empty to process all
TARGET_PROPERTIES = [
    # "P569",  # Date of Birth
    # "P570",  # Date of Death
    # "P21",  # Sex or Gender
]
REPORT_HISTORY_DEPTH = 20  # Revision pairs scanned per report page
QID_PATTERN = re.compile(r"\[\[(Q\d+)\]\]")
INVALID_REPORT_SECTIONS = {
    "Types statistics",
}

REPORT_VIOLATION_QID_PATTERN = re.compile(r"Q\|?(\d+)")
QID_EXACT_PATTERN = re.compile(r"^Q\d+$")
PID_EXACT_PATTERN = re.compile(r"^P\d+$")


def is_qid(value):
    """Return True if the value looks like a Wikidata item id (Q*)."""
    if not isinstance(value, str):
        return False
    return bool(QID_EXACT_PATTERN.fullmatch(value.strip()))


def is_pid(value):
    """Return True if the value looks like a Wikidata property id (P*)."""
    if not isinstance(value, str):
        return False
    return bool(PID_EXACT_PATTERN.fullmatch(value.strip()))


def is_entity_or_property_id(value):
    """Return True for valid QIDs or PIDs."""
    return is_qid(value) or is_pid(value)


def is_valid_violation_section(section):
    """Return True if a report section represents a real violation bucket."""
    if not section:
        return False
    if section in INVALID_REPORT_SECTIONS:
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


SITE = None

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_CACHE_FILE = CACHE_DIR / "id_labels_en.json"
REPAIR_CANDIDATES_FILE = DATA_DIR / "01_repair_candidates.json"
WIKIDATA_REPAIRS = DATA_DIR / "02_wikidata_repairs.json"

RUN_ID = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
STATS_FILE = LOG_DIR / f"fetcher_stats_{RUN_ID}.jsonl"
SUMMARY_FILE = LOG_DIR / f"run_summary_{RUN_ID}.json"


def get_wikidata_site():
    """Lazy initializer for the MediaWiki client used during candidate mining."""
    if mwclient is None:
        raise RuntimeError(
            "mwclient is required to build 01_repair_candidates.json. Install it via 'pip install mwclient'."
        )
    global SITE
    if SITE is None:
        SITE = mwclient.Site("www.wikidata.org", clients_useragent=HEADERS["User-Agent"])
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
        qids = QID_PATTERN.findall(line)
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


def ensure_repair_candidates_file(filename, history_limit=REPORT_HISTORY_DEPTH):
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

    global TARGET_PROPERTIES
    if not TARGET_PROPERTIES:
        print("[*] No TARGET_PROPERTIES defined. Auto-discovering from summary page...")
        TARGET_PROPERTIES = fetch_all_active_properties()
    if not TARGET_PROPERTIES:
        print("[!] Failed to identify any properties to mine.")
        return []

    # Rebuild candidate list
    print(f"[!] {filename} missing. Mining fresh candidate list...")
    fresh_candidates = []
    for prop in TARGET_PROPERTIES:
        fresh_candidates.extend(mine_repairs(prop, max_items=history_limit))

    # Save to disk
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(fresh_candidates, fh, indent=2)
    print(f"[+] Done. Found {len(fresh_candidates)} candidates. Saved to {filename}.")
    return fresh_candidates


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


class StatsLogger:
    """Append-only JSONL logger for per-candidate fetch diagnostics."""

    def __init__(self, stats_path):
        """Initialize logger with target file and shared run_id."""
        self.stats_path = stats_path
        self.run_id = RUN_ID

    def log(self, record):
        """Write a single JSON object line enriched with the run identifier."""
        enriched = {"run_id": self.run_id}
        enriched.update(record)
        with open(self.stats_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(enriched, ensure_ascii=True))
            fh.write("\n")


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


MISSING_LABEL_PLACEHOLDER = "Label unavailable"


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


REQUIRED_WORLD_STATE_KEYS = ("L1_ego_node", "L2_labels", "L3_neighborhood", "L4_constraints")


def _ensure(condition, message):
    if not condition:
        raise ValueError(message)


def validate_world_state_entry(entry_id, entry, valid_ids=None):
    """Validate a single world state entry against the Stage-3 contract."""
    _ensure(isinstance(entry_id, str), f"World state key {entry_id!r} must be a string id.")
    if valid_ids is not None:
        _ensure(
            entry_id in valid_ids,
            f"World state key {entry_id!r} not found in Stage-2 repair ids.",
        )
    _ensure(isinstance(entry, dict), f"World state entry for {entry_id} must be an object.")
    for required in REQUIRED_WORLD_STATE_KEYS:
        _ensure(required in entry, f"World state entry {entry_id} missing required layer {required}.")
    ego = entry["L1_ego_node"]
    _ensure(isinstance(ego, dict), f"L1_ego_node for {entry_id} must be an object.")
    for field in ("qid", "label", "description", "properties"):
        _ensure(field in ego, f"L1_ego_node for {entry_id} missing field {field}.")
    _ensure(isinstance(ego["properties"], dict), f"L1_ego_node properties for {entry_id} must be an object.")
    neighborhood = entry["L3_neighborhood"]
    _ensure(isinstance(neighborhood, dict), f"L3_neighborhood for {entry_id} must be a dict.")
    _ensure(
        "outgoing_edges" in neighborhood,
        f"L3_neighborhood for {entry_id} missing outgoing_edges.",
    )
    edges = neighborhood["outgoing_edges"]
    _ensure(isinstance(edges, list), f"L3_neighborhood outgoing_edges for {entry_id} must be a list.")
    for idx, edge in enumerate(edges):
        _ensure(isinstance(edge, dict), f"Edge #{idx} for {entry_id} must be an object.")
        for field in ("property_id", "target_qid", "target_label", "target_description"):
            _ensure(field in edge, f"Edge #{idx} for {entry_id} missing field {field}.")
    _ensure(isinstance(entry["L2_labels"], dict), f"L2_labels for {entry_id} must be a dict.")
    _ensure(isinstance(entry["L4_constraints"], dict), f"L4_constraints for {entry_id} must be a dict.")


def validate_world_state_document(world_state, valid_ids):
    """Ensure the top-level structure matches the Stage-3 requirements."""
    _ensure(isinstance(world_state, dict), "World state root must be a JSON object.")
    valid_ids = set(valid_ids or [])
    keys = set(world_state.keys())
    missing = valid_ids - keys
    unexpected = keys - valid_ids
    _ensure(
        not missing,
        f"World state missing {len(missing)} ids from Stage-2 dataset: {sorted(list(missing))[:5]} ...",
    )
    _ensure(
        not unexpected,
        f"World state has unexpected ids not found in Stage-2 dataset: {sorted(list(unexpected))[:5]} ...",
    )
    for entry_id, entry in world_state.items():
        validate_world_state_entry(entry_id, entry, valid_ids)


def extract_repair_ids(dataset):
    """Return ordered list of repair ids extracted from Stage-2 dataset entries."""
    ids = []
    valid_entry_count = 0
    for entry in dataset or []:
        if not isinstance(entry, dict):
            continue
        valid_entry_count += 1
        entry_id = entry.get("id")
        if entry_id:
            ids.append(entry_id)
    return ids, valid_entry_count


def ensure_unique_ids(ids):
    """Validate that ids are unique and return the deduplicated set."""
    seen = set()
    duplicates = set()
    for entry_id in ids:
        if entry_id in seen:
            duplicates.add(entry_id)
        else:
            seen.add(entry_id)
    if duplicates:
        raise ValueError(f"Duplicate Stage-2 ids detected: {sorted(list(duplicates))[:5]} ...")
    return seen


def ensure_all_entries_have_ids(total_entries, ids):
    """Ensure that every Stage-2 dataset entry contributed an id."""
    missing = total_entries - len(ids)
    if missing > 0:
        raise ValueError(f"{missing} Stage-2 entries are missing an 'id' field.")


def validate_world_state_file(world_state_path, repairs_path):
    """Validate an on-disk world state artifact against Stage-2 repairs."""
    dataset = load_cached_repairs(repairs_path)
    if dataset is None:
        raise RuntimeError(f"Stage-2 repairs file missing or empty at {repairs_path}.")
    repair_ids, total_entries = extract_repair_ids(dataset)
    ensure_all_entries_have_ids(total_entries, repair_ids)
    expected_ids = ensure_unique_ids(repair_ids)
    path = Path(world_state_path)
    if not path.exists():
        raise RuntimeError(f"World state file not found at {world_state_path}.")
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    validate_world_state_document(payload, expected_ids)
    print(f"[+] World state validation passed for {len(payload)} entries.")


class WorldStateBuilder:
    """Streams the 2025 dump to attach ego/neighbor context to each repair."""

    def __init__(self, dump_path):
        """Record dump location and flag whether it is available."""
        self.dump_path = Path(dump_path)
        self.has_dump = self.dump_path.exists()

    def build(self, entries):
        """Yield world_state payloads for the provided repair entries."""
        if not entries:
            return
        if not self.has_dump:
            print(f"[!] Context Builder skipped: dump not found at {self.dump_path}")
            return

        # 1. Identify Initial Targets (Focus + Property)
        focus_ids = {entry["qid"] for entry in entries}
        property_ids = {entry["property"] for entry in entries}
        target_ids = focus_ids | property_ids

        print(f"[*] Context Builder: Single-pass stream for {len(target_ids)} primary entities...")

        # 2. Single Pass Scan
        # We only load the Focus and Property entities from the dump.
        retrieved_entities = self._load_entities_from_dump(target_ids)

        focus_entities = {eid: retrieved_entities[eid] for eid in focus_ids if eid in retrieved_entities}
        property_entities = {pid: retrieved_entities[pid] for pid in property_ids if pid in retrieved_entities}

        missing_focus = focus_ids - set(focus_entities)
        if missing_focus:
            print(f"    [!] Warning: {len(missing_focus)} focus entities not found in dump.")

        # 3. Identify Neighbors from the loaded data
        neighbor_ids = self._collect_neighbor_targets(focus_entities.values())
        constraint_target_ids = self._collect_constraint_targets(property_entities)

        # Filter out neighbors we already happened to load (if they were focus nodes too)
        preloaded_ids = set(retrieved_entities.keys())
        api_label_targets = (neighbor_ids | constraint_target_ids) - preloaded_ids

        print(
            f"[*] Context Builder: Fetching {len(api_label_targets)} neighbor/constraint labels via API (avoiding 2nd dump scan)..."
        )

        # 4. API Fallback for Neighbors + constraint references (Much faster than 2nd scan)
        # We fetch these just for labels/descriptions
        neighbor_entities_api = self._fetch_labels_via_api(api_label_targets)

        # Merge all available entity data for lookup
        # (Prioritize Dump data, fall back to API data)
        full_entity_map = {**neighbor_entities_api, **retrieved_entities}

        for entry in tqdm(entries, desc="Building world states", unit="entry"):
            focus_entity = focus_entities.get(entry["qid"])
            if not focus_entity:
                continue

            property_entity = property_entities.get(entry["property"])

            context = self._assemble_world_state(
                focus_entity,
                full_entity_map,  # Pass the full map so we can find neighbors
                entry,
                property_entity,
            )
            yield entry["id"], context

    def _load_entities_from_dump(self, target_ids):
        """Single-pass stream over the dump extracting matching entity blobs."""
        found = {}
        if not target_ids or not self.has_dump:
            return found
        target_ids = set(target_ids)

        try:
            with gzip.open(self.dump_path, "rb") as fh:
                stream = ijson.items(fh, "item")
                for entity in tqdm(
                    stream,
                    desc="Scanning dump for context",
                    unit=" entity",
                    miniters=10000,
                    total=118319831,
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:,}/{total:,}{unit} [{elapsed}<{remaining}, {rate_fmt}]",
                ):
                    eid = entity.get("id")
                    if eid in target_ids:
                        found[eid] = entity
                        # Stop early if we found everything
                        if len(found) == len(target_ids):
                            tqdm.write("    [+] Found all targets. Stopping stream early.")
                            break
        except Exception as exc:
            print(f"    [!] Dump stream error: {exc}")
        return found

    def _collect_neighbor_targets(self, entities):
        """Gather unique QIDs reachable via outbound statements."""
        neighbors = set()
        for entity in tqdm(entities, desc="Collecting neighbors", unit="focus"):
            edges = 0
            claims = entity.get("claims", {})
            for pid, statements in claims.items():
                for claim in statements:
                    if edges >= MAX_NEIGHBOR_EDGES:
                        break
                    datavalue = claim.get("mainsnak", {}).get("datavalue")
                    if not datavalue:
                        continue
                    value = datavalue.get("value")
                    if isinstance(value, dict) and value.get("entity-type") in {"item", "property"}:
                        target_id = value.get("id")
                        if target_id:
                            neighbors.add(target_id)
                            edges += 1
                if edges >= MAX_NEIGHBOR_EDGES:
                    break
        return neighbors

    def _collect_constraint_targets(self, property_entities):
        """Gather constraint-related entity/property IDs for downstream label lookups."""
        targets = set()
        if not property_entities:
            return targets
        for entity in property_entities.values():
            constraint_claims = entity.get("claims", {}).get("P2302", [])
            for claim in constraint_claims:
                snak = claim.get("mainsnak", {})
                datavalue = snak.get("datavalue")
                if datavalue:
                    value = datavalue.get("value")
                    if isinstance(value, dict):
                        constraint_qid = value.get("id")
                        if constraint_qid:
                            targets.add(constraint_qid)
                qualifiers = claim.get("qualifiers", {})
                for qualifier_pid, qualifier_values in qualifiers.items():
                    if qualifier_pid:
                        targets.add(qualifier_pid)
                    for qualifier in qualifier_values:
                        qualifier_value = qualifier.get("datavalue", {}).get("value")
                        if isinstance(qualifier_value, dict):
                            entity_type = qualifier_value.get("entity-type")
                            qualifier_qid = qualifier_value.get("id")
                            if entity_type in {"item", "property"} and qualifier_qid:
                                targets.add(qualifier_qid)
        return targets

    def _fetch_labels_via_api(self, ids):
        """Resolve missing neighbor labels/descriptions via Action API."""
        resolved = {}
        id_list = list(ids)
        if not id_list:
            return resolved
        for start in tqdm(
            range(0, len(id_list), 50),
            desc="Fetching neighbor labels",
            unit="batch",
        ):
            batch = id_list[start : start + 50]
            params = {
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "props": "labels|descriptions",
            }
            data = get_json(params)
            if not data or "entities" not in data:
                continue
            for entity_id, entity in data["entities"].items():
                if not entity or "missing" in entity:
                    continue
                resolved[entity_id] = {
                    "id": entity_id,
                    "labels": entity.get("labels", {}),
                    "descriptions": entity.get("descriptions", {}),
                }
        return resolved

    def _assemble_world_state(self, focus_entity, full_entity_map, entry, property_entity):
        """Return the world_state JSON object for one repair entry."""
        property_id = entry["property"]
        focus_node = {
            "qid": focus_entity.get("id"),
            "label": pick_label(focus_entity),
            "description": pick_description(focus_entity),
            "sitelinks_count": len(focus_entity.get("sitelinks", {})),
            "properties": self._extract_properties(focus_entity),
        }
        neighborhood_snapshot = self._build_neighborhood_snapshot(focus_entity, full_entity_map)
        constraint_metadata = self._extract_constraints(property_id, property_entity, full_entity_map)
        label_layer = self._build_label_layer(
            focus_entity,
            property_id,
            property_entity,
            neighborhood_snapshot,
            constraint_metadata,
            full_entity_map,
        )
        world_state = {
            "L1_ego_node": focus_node,
            "L2_labels": label_layer,
            "L3_neighborhood": neighborhood_snapshot,
            "L4_constraints": constraint_metadata,
        }
        if entry.get("track") == "T_BOX":
            constraint_context = self._build_constraint_change_context(entry)
            if constraint_context:
                world_state["constraint_change_context"] = constraint_context
        return world_state

    def _extract_properties(self, entity):
        """Collect up to MAX_PROPERTY_VALUES per property for the ego node."""
        properties = {}
        claims = entity.get("claims", {})
        for pid, statements in claims.items():
            values = []
            for claim in statements:
                snak = claim.get("mainsnak", {})
                if snak.get("snaktype") != "value":
                    continue
                datavalue = snak.get("datavalue")
                if not datavalue:
                    continue
                values.append(format_datavalue(datavalue.get("value")))
                if len(values) >= MAX_PROPERTY_VALUES:
                    break
            if values:
                properties[pid] = values
        return properties

    def _build_neighborhood_snapshot(self, entity, neighbor_entities):
        """Capture labeled 1-hop outgoing edges for Type B reasoning tests."""
        edges = []
        edge_count = 0
        claims = entity.get("claims", {})
        for pid, statements in claims.items():
            for claim in statements:
                if edge_count >= MAX_NEIGHBOR_EDGES:
                    break
                snak = claim.get("mainsnak", {})
                datavalue = snak.get("datavalue")
                if not datavalue:
                    continue
                value = datavalue.get("value")
                if isinstance(value, dict) and value.get("entity-type") in {"item", "property"}:
                    target_id = value.get("id")
                    neighbor = neighbor_entities.get(target_id)
                    edges.append(
                        {
                            "property_id": pid,
                            "target_qid": target_id,
                            "target_label": pick_label(neighbor),
                            "target_description": pick_description(neighbor),
                        }
                    )
                    edge_count += 1
            if edge_count >= MAX_NEIGHBOR_EDGES:
                break
        return {"outgoing_edges": edges}

    def _build_label_layer(
        self,
        focus_entity,
        property_id,
        property_entity,
        neighborhood_snapshot,
        constraint_metadata,
        entity_map,
    ):
        """Construct the explicit L2 label."""
        label_index = {}

        def track(entity_id):
            if not entity_id or entity_id in label_index:
                return
            entity = entity_map.get(entity_id)
            label = pick_label(entity) if entity else None
            description = pick_description(entity) if entity else None
            label_index[entity_id] = {
                "label": label or MISSING_LABEL_PLACEHOLDER,
                "description": description,
            }

        if focus_entity:
            track(focus_entity.get("id"))
        track(property_id)
        if property_entity:
            track(property_entity.get("id"))
        for edge in neighborhood_snapshot.get("outgoing_edges", []):
            track(edge.get("property_id"))
            track(edge.get("target_qid"))
        constraint_layer = constraint_metadata or {}
        track(constraint_layer.get("property_id"))
        for constraint in constraint_layer.get("constraints", []):
            constraint_type = constraint.get("constraint_type", {})
            track(constraint_type.get("qid"))
            for qualifier in constraint.get("qualifiers", []):
                track(qualifier.get("property_id"))
                for value in qualifier.get("values", []):
                    track(value.get("qid"))

        return {"entities": label_index}

    def _lookup_label(self, entity_id, entity_map):
        """Return the preferred label for entity_id or a placeholder if missing."""
        if not entity_id:
            return None
        entity = entity_map.get(entity_id)
        label = pick_label(entity)
        return label or MISSING_LABEL_PLACEHOLDER

    def _extract_constraints(self, property_id, property_entity, full_entity_map):
        """Summarize the on-wiki constraint definition (P2302 statements) with labels."""
        if not property_entity:
            return {"property_id": property_id, "constraints": []}
        constraints = []
        constraint_claims = property_entity.get("claims", {}).get("P2302", [])
        for claim in constraint_claims:
            snak = claim.get("mainsnak", {})
            datavalue = snak.get("datavalue")
            constraint_qid = None
            if datavalue:
                value = datavalue.get("value")
                if isinstance(value, dict):
                    constraint_qid = value.get("id")
            qualifiers = claim.get("qualifiers", {})
            qualifier_parts = []
            qualifier_details = []
            for qualifier_pid, qualifier_values in qualifiers.items():
                property_label = self._lookup_label(qualifier_pid, full_entity_map) or MISSING_LABEL_PLACEHOLDER
                rendered_values = []
                qualifier_summary_values = []
                for qualifier in qualifier_values:
                    dv = qualifier.get("datavalue")
                    if not dv:
                        continue
                    raw_value = format_datavalue(dv.get("value"))
                    rendered_entry = {"raw": raw_value}
                    qualifier_value = dv.get("value")
                    qualifier_qid = None
                    if isinstance(qualifier_value, dict):
                        entity_type = qualifier_value.get("entity-type")
                        if entity_type in {"item", "property"}:
                            qualifier_qid = qualifier_value.get("id")
                    if qualifier_qid:
                        rendered_entry["qid"] = qualifier_qid
                        rendered_entry["label"] = self._lookup_label(qualifier_qid, full_entity_map)
                        qualifier_summary_values.append(f"{rendered_entry['label']} ({qualifier_qid})")
                    else:
                        qualifier_summary_values.append(raw_value)
                    rendered_values.append(rendered_entry)
                qualifier_details.append(
                    {
                        "property_id": qualifier_pid,
                        "property_label": property_label,
                        "values": rendered_values,
                    }
                )
                if rendered_values:
                    qualifier_parts.append(
                        f"{property_label} ({qualifier_pid}): {', '.join(qualifier_summary_values)}"
                    )
            summary = "; ".join(qualifier_parts) if qualifier_parts else "No qualifiers recorded."
            constraint_label = self._lookup_label(constraint_qid, full_entity_map) if constraint_qid else None
            constraints.append(
                {
                    "constraint_type": {
                        "qid": constraint_qid,
                        "label": constraint_label or (MISSING_LABEL_PLACEHOLDER if constraint_qid else None),
                    },
                    "qualifiers": qualifier_details,
                    "rule_summary": summary,
                }
            )
        return {
            "property_id": property_id,
            "constraints": constraints,
        }

    def _build_constraint_change_context(self, entry):
        """Return constraint before/after metadata for T-box records."""
        repair_target = entry.get("repair_target") or {}
        if repair_target.get("kind") != "T_BOX":
            return None
        delta = repair_target.get("constraint_delta") or {}
        context = {
            "property_revision_id": repair_target.get("property_revision_id"),
            "property_revision_prev": repair_target.get("property_revision_prev"),
            "signatures": {
                "before": {
                    "hash": delta.get("hash_before"),
                    "signature": delta.get("signature_before"),
                    "signature_raw": delta.get("signature_before_raw"),
                },
                "after": {
                    "hash": delta.get("hash_after"),
                    "signature": delta.get("signature_after"),
                    "signature_raw": delta.get("signature_after_raw"),
                },
            },
        }
        if delta.get("changed_constraint_types") is not None:
            context["changed_constraint_types"] = delta.get("changed_constraint_types")
        if delta.get("old_constraints") is not None:
            context["constraints_before"] = delta["old_constraints"]
        if delta.get("new_constraints") is not None:
            context["constraints_after"] = delta["new_constraints"]
        return context


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
    start_dt = end_dt - timedelta(days=REVISION_LOOKBACK_DAYS)
    return format_timestamp(start_dt), format_timestamp(end_dt)


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


def get_json(params=None, *, endpoint=API_ENDPOINT, with_format=True):
    """Wrapper around requests.get with retries and default MediaWiki params."""
    query = dict(params or {})
    if with_format:
        query.setdefault("format", "json")
        query.setdefault("formatversion", 2)
    for attempt in range(4):
        try:
            response = requests.get(
                endpoint,
                headers=HEADERS,
                params=query if query else None,
                timeout=API_TIMEOUT,
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
        time.sleep(0.2)
    return None


class LabelResolver:
    """Deterministic ID -> label resolution with disk caching."""

    def __init__(self, cache_path=LABEL_CACHE_FILE, preferred_lang="en"):
        self.cache_path = Path(cache_path)
        self.preferred_lang = preferred_lang
        self.cache = {}
        self.failed_ids = set()
        self._load_cache()

    def _load_cache(self):
        if not self.cache_path.exists():
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            print(f"[!] Warning: Could not load label cache {self.cache_path}: {exc}")
            return
        if isinstance(payload, dict):
            self.cache.update(payload)

    def _persist_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        ordered_keys = sorted(self.cache.keys())
        serialized = {key: self.cache[key] for key in ordered_keys}
        with open(self.cache_path, "w", encoding="utf-8") as fh:
            json.dump(serialized, fh, ensure_ascii=True, indent=2, sort_keys=True)

    @staticmethod
    def _null_entry():
        return {
            "label_en": None,
            "description_en": None,
        }

    def resolve(self, ids):
        """Resolve a batch of ids and return {id: resolution}."""
        if not ids:
            return {}
        ordered_ids = []
        seen = set()
        for entity_id in ids:
            if not is_entity_or_property_id(entity_id):
                continue
            if entity_id in seen:
                continue
            seen.add(entity_id)
            ordered_ids.append(entity_id)
        if not ordered_ids:
            return {}
        missing = [entity_id for entity_id in ordered_ids if entity_id not in self.cache]
        if missing:
            self._fetch_and_cache(missing)
        return {entity_id: self.cache.get(entity_id, self._null_entry()) for entity_id in ordered_ids}

    def lookup(self, entity_id):
        """Return cached resolution for a single id."""
        if not is_entity_or_property_id(entity_id):
            return self._null_entry()
        if entity_id not in self.cache:
            self.resolve([entity_id])
        return self.cache.get(entity_id, self._null_entry())

    def _fetch_and_cache(self, ids):
        newly_cached = False
        for batch in chunked(ids, 50):
            params = {
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "props": "labels|descriptions",
            }
            data = get_json(params)
            resolved_ids = set()
            if data and "entities" in data:
                for entity_id, entity in data["entities"].items():
                    resolved_ids.add(entity_id)
                    if not entity or "missing" in entity:
                        self.cache[entity_id] = self._null_entry()
                        continue
                    self.cache[entity_id] = {
                        "label_en": pick_label(entity, self.preferred_lang),
                        "description_en": pick_description(entity, self.preferred_lang),
                    }
                    newly_cached = True
            unresolved = set(batch) - resolved_ids
            for missing_id in unresolved:
                if missing_id in self.failed_ids:
                    continue
                print(f"    [!] Warning: Unable to resolve {missing_id} via Wikidata API.")
                self.failed_ids.add(missing_id)
                self.cache[missing_id] = self._null_entry()
                newly_cached = True
        if newly_cached:
            self._persist_cache()


def get_current_state(qid, property_id):
    """Fetch today's claim values for persistence checking."""
    params = {
        "action": "wbgetentities",
        "ids": qid,
        "props": "claims",
    }
    data = get_json(params)
    if not data or "entities" not in data:
        return None
    entity = data["entities"].get(qid)
    if not entity or "missing" in entity:
        return None
    claims = entity.get("claims", {}).get(property_id, [])
    signature, values = summarize_claims(claims)
    if signature == ("MISSING",):
        return None
    return values


def fetch_revision_history(qid, start_time, end_time):
    """Collect revision metadata within [start_time, end_time] from REST history."""
    start_dt = parse_iso8601(start_time) if start_time else None
    end_dt = parse_iso8601(end_time) if end_time else None
    revisions = []
    carry_revision = None
    endpoint = REST_HISTORY_URL.format(qid=qid)
    next_endpoint = endpoint
    params = {"limit": 200}
    batches = 0
    truncated_by_window = False
    reached_page_limit = False
    api_calls = 0

    while next_endpoint and batches < MAX_HISTORY_PAGES:
        data = get_json(
            params=params if next_endpoint == endpoint else None,
            endpoint=next_endpoint,
            with_format=False,
        )
        if not data or "revisions" not in data:
            break
        api_calls += 1
        for rev in data.get("revisions", []):
            rev_ts = rev.get("timestamp")
            rev_dt = parse_iso8601(rev_ts)
            if not rev_dt:
                continue
            if end_dt and rev_dt > end_dt:
                continue
            if start_dt and rev_dt < start_dt:
                truncated_by_window = True
                if not carry_revision:
                    carry_revision = rev
                next_endpoint = None
                break
            revisions.append(rev)
        if next_endpoint is None:
            break
        older_url = data.get("older")
        if not older_url:
            break
        next_endpoint = older_url
        batches += 1
        params = None
        if start_dt and revisions:
            oldest_dt = parse_iso8601(revisions[-1]["timestamp"])
            if oldest_dt and oldest_dt < start_dt:
                truncated_by_window = True
                break
    if carry_revision:
        revisions.append(carry_revision)
    revisions.sort(key=lambda rev: rev["timestamp"])
    if next_endpoint and batches >= MAX_HISTORY_PAGES:
        reached_page_limit = True

    history_meta = {
        "qid": qid,
        "start_time": start_time,
        "end_time": end_time,
        "lookback_days": REVISION_LOOKBACK_DAYS,
        "max_history_pages": MAX_HISTORY_PAGES,
        "api_calls": api_calls,
        "batches_used": batches,
        "revisions_scanned": len(revisions),
        "earliest_revision": revisions[0]["timestamp"] if revisions else None,
        "latest_revision": revisions[-1]["timestamp"] if revisions else None,
        "truncated_by_window": truncated_by_window,
        "reached_page_limit": reached_page_limit,
        "carry_revision_used": carry_revision is not None,
        "truncated": truncated_by_window or reached_page_limit,
    }

    return revisions, history_meta


def get_claims_for_revision(qid, property_id, revision_id):
    """Fetch claims for a specific revision via Special:EntityData snapshot."""
    endpoint = ENTITY_DATA_URL.format(qid=qid)
    data = get_json(
        params={"revision": revision_id},
        endpoint=endpoint,
        with_format=False,
    )
    if not data or "entities" not in data:
        return []
    entity = data["entities"].get(qid)
    if not entity or "missing" in entity:
        return []
    return entity.get("claims", {}).get(property_id, [])


def extract_user(revision):
    """Normalize revision user data regardless of REST/Action response shape."""
    user = revision.get("user")
    if isinstance(user, dict):
        return user.get("name", "unknown")
    return user or "unknown"


def find_repair_revision(qid, property_id, start_time, end_time):
    """Walk revision history to locate the first diff that changes the property."""
    revisions, history_meta = fetch_revision_history(qid, start_time, end_time)
    if not revisions:
        return None, history_meta

    previous_signature = None
    previous_snapshot = None

    last_valid_repair = None
    for rev in revisions:
        revision_id = rev.get("id") or rev.get("revid")
        if not revision_id:
            continue
        current_claims = get_claims_for_revision(qid, property_id, revision_id)
        current_signature, current_snapshot = summarize_claims(current_claims)
        if previous_signature is not None and current_signature != previous_signature:
            # Found a change. Update our candidate, but keep looking.
            last_valid_repair = {
                "repair_revision_id": revision_id,
                "timestamp": rev.get("timestamp"),
                "action": classify_action(previous_signature, current_signature),
                "old_value": previous_snapshot,
                "new_value": current_snapshot,
                "author": extract_user(rev),
            }

        previous_signature = current_signature
        previous_snapshot = current_snapshot

    return last_valid_repair, history_meta


def find_tbox_reform_revision(
    property_id,
    start_time,
    end_time,
    *,
    include_snapshots=True,
    scan_from_end=False,
    max_revisions=None,
):
    """
    Inspect the property entity history to locate constraint (P2302) changes.
    This captures T-box edits where the schema itself was modified instead of the instance data.
    When scan_from_end=True we iterate from newest revisions backwards and stop on the first change detected,
    which is useful for ambiguity probes that only need a boolean answer.
    """
    property_title = f"Property:{property_id}"
    revisions, history_meta = fetch_revision_history(property_title, start_time, end_time)
    if not revisions:
        return None, history_meta

    ordered_revisions = list(revisions)

    if scan_from_end:
        # Cheap ambiguity path: walk newest->oldest, stop after first detected change or max_revisions.
        next_signature = None
        next_revision = None
        scanned = 0
        for rev in reversed(ordered_revisions):
            if max_revisions is not None and scanned >= max_revisions:
                break
            revision_id = rev.get("id") or rev.get("revid")
            if not revision_id:
                continue
            constraint_claims = get_claims_for_revision(property_id, "P2302", revision_id)
            current_signature = signature_p2302(constraint_claims)
            if next_signature and current_signature["hash"] != next_signature["hash"]:
                delta = _build_constraint_delta(current_signature, next_signature, include_snapshots)
                tbox_event = {
                    "property_revision_id": (next_revision.get("id") or next_revision.get("revid"))
                    if next_revision
                    else None,
                    "property_revision_prev": revision_id,
                    "timestamp": next_revision.get("timestamp") if next_revision else rev.get("timestamp"),
                    "author": extract_user(next_revision or rev),
                    "constraint_delta": delta,
                }
                return tbox_event, history_meta
            next_signature = current_signature
            next_revision = rev
            scanned += 1
        return None, history_meta

    previous_signature = None
    previous_revision_id = None
    last_change = None
    scanned = 0

    for rev in ordered_revisions:
        if max_revisions is not None and scanned >= max_revisions:
            break
        revision_id = rev.get("id") or rev.get("revid")
        if not revision_id:
            continue
        constraint_claims = get_claims_for_revision(property_id, "P2302", revision_id)
        current_signature = signature_p2302(constraint_claims)

        if previous_signature and current_signature["hash"] != previous_signature["hash"]:
            # Track the most recent constraint change so we align with the report disappearance.
            delta = _build_constraint_delta(previous_signature, current_signature, include_snapshots)
            last_change = {
                "property_revision_id": revision_id,
                "property_revision_prev": previous_revision_id,
                "timestamp": rev.get("timestamp"),
                "author": extract_user(rev),
                "constraint_delta": delta,
            }

        previous_signature = current_signature
        previous_revision_id = revision_id
        scanned += 1

    return last_change, history_meta


def parse_report_violation_type_qids(raw_text):
    """Extract normalized QIDs from the raw violation type string."""
    if not raw_text or not isinstance(raw_text, str):
        return []
    seen = set()
    ordered = []
    for match in REPORT_VIOLATION_QID_PATTERN.finditer(raw_text):
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


def add_resolved_list_fields(container, field_name, ids, resolved_lookup):
    """Attach *_labels_en/_descriptions_en based on an id list."""
    if not container or not ids:
        return
    labels = []
    descriptions = []
    for entity_id in ids:
        resolution = resolved_lookup.get(entity_id) or LabelResolver._null_entry()
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
        constraint_resolution = resolved_lookup.get(constraint_qid) or LabelResolver._null_entry()
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
            property_resolution = resolved_lookup.get(qualifier_pid) or LabelResolver._null_entry()
            parameter_key = qualifier_pid
            parameter_values = []
            summary_values = []
            for raw_value in qualifier.get("values") or []:
                if is_qid(raw_value) or is_pid(raw_value):
                    value_resolution = resolved_lookup.get(raw_value) or LabelResolver._null_entry()
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
    violation_value_current_qids = extract_qids_from_sequence(violation_context.get("value_current_2025"))
    ids_to_resolve.update(violation_value_qids)
    ids_to_resolve.update(violation_value_current_qids)

    persistence_qids = extract_qids_from_sequence(persistence_check.get("current_value_2025"))
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
            return LabelResolver._null_entry()
        return resolved_lookup.get(entity_id) or LabelResolver._null_entry()

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
    add_resolved_list_fields(violation_context, "value_current_2025", violation_value_current_qids, resolved_lookup)
    add_resolved_list_fields(persistence_check, "current_value_2025", persistence_qids, resolved_lookup)
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


def process_pipeline(max_candidates=None):
    """Main entry point: reads candidates, finds repairs, and builds context."""
    input_file = REPAIR_CANDIDATES_FILE
    candidates = ensure_repair_candidates_file(input_file)
    if not candidates:
        print(f"[!] Unable to proceed without {input_file}.")
        return

    label_resolver = LabelResolver()
    dataset = load_cached_repairs(WIKIDATA_REPAIRS)
    seen_ids = set()
    if dataset:
        for entry in dataset:
            if isinstance(entry, dict):
                entry_id = entry.get("id")
                if entry_id:
                    seen_ids.add(entry_id)

    summary = None
    if dataset is not None:
        print(
            "[*] Using cached repairs file for Stage 3. Delete data/02_wikidata_repairs.json to force recompute Stage 2."
        )
        enrich_repair_entries(dataset, label_resolver)
        with open(WIKIDATA_REPAIRS, "w", encoding="utf-8") as out:
            json.dump(dataset, out, indent=2)
    else:
        stats_logger = StatsLogger(STATS_FILE)
        summary = {
            "run_id": stats_logger.run_id,
            "lookback_days": REVISION_LOOKBACK_DAYS,
            "max_history_pages": MAX_HISTORY_PAGES,
            "total_candidates": len(candidates),
            "processed": 0,
            "persistence_failed": 0,
            "bad_fix_date": 0,
            "repairs_found": 0,
            "repairs_found_a_box": 0,
            "repairs_found_t_box": 0,
            "ambiguous_both_changed": 0,
            "no_diff": 0,
            "no_history": 0,
            "truncated_by_window": 0,
            "reached_page_limit": 0,
            "duplicates_skipped": 0,
        }

        def record_history_stats(meta):
            """Increment window/page limit counters for any history call."""
            if not meta:
                return
            if meta.get("truncated_by_window"):
                summary["truncated_by_window"] += 1
            if meta.get("reached_page_limit"):
                summary["reached_page_limit"] += 1

        def history_scanned(meta):
            """Return True if the REST history call yielded any revisions."""
            return bool(meta and meta.get("revisions_scanned", 0) > 0)

        print(f"[*] Loaded {len(candidates)} candidates. Using REST history.")

        dataset = []
        seen_ids.clear()
        total_to_process = len(candidates)
        if max_candidates is not None:
            total_to_process = min(total_to_process, max_candidates)
        progress = tqdm(total=total_to_process, desc="Processing candidates", unit="candidate")
        for i, item in enumerate(candidates):
            if i >= total_to_process:
                break
            qid = item["qid"]
            pid = item["property_id"]
            violation_type = item.get("violation_type")
            violation_type_normalized = normalize_report_violation_type(violation_type)

            if not qid.startswith("Q"):
                progress.update(1)
                continue

            if TARGET_PROPERTIES and pid not in TARGET_PROPERTIES:
                progress.update(1)
                continue

            progress.write(f"[{i + 1}/{total_to_process}] Analyzing {qid} ({pid})...")
            summary["processed"] += 1

            record_base = {
                "qid": qid,
                "property": pid,
                "violation_type": violation_type,
            }
            report_metadata = build_report_provenance(item, pid)

            report_date = item["fix_date"]
            start_time, end_time = compute_revision_window(report_date)
            if not end_time:
                progress.write("    [x] Dropped: Could not parse fix_date.")
                summary["bad_fix_date"] += 1
                stats_logger.log(
                    {
                        **record_base,
                        "result": "bad_fix_date",
                        "report_date": report_date,
                    }
                )
                progress.update(1)
                continue

            fix_event, history_meta = find_repair_revision(
                qid,
                pid,
                start_time=start_time,
                end_time=end_time,
            )
            tbox_event = None
            tbox_history_meta = None
            ambiguous_history_meta = None

            # Cleaner (A-box) path: prefer instance-level repairs when they exist.
            if fix_event:
                progress.write(f"    [+] FOUND A-BOX REPAIR! {fix_event['old_value']} -> {fix_event['new_value']}")
                summary["repairs_found"] += 1
                summary["repairs_found_a_box"] += 1
                current_values_live = get_current_state(qid, pid)
                if current_values_live is None and STRICT_PERSISTENCE and fix_event["action"] != "DELETE":
                    progress.write("    [x] Dropped: Persistence check failed (Entity/Prop missing).")
                    summary["persistence_failed"] += 1
                    stats_logger.log(
                        {
                            **record_base,
                            "result": "persistence_failed",
                            "reason": "missing_current_value",
                            "track": "A_BOX",
                            "repair_revision_id": fix_event["repair_revision_id"],
                            "action": fix_event["action"],
                        }
                    )
                    progress.update(1)
                    continue
                normalized_current_values = current_values_live if current_values_live is not None else []
                violation_context = {
                    "report_violation_type": violation_type,
                    "value": fix_event["old_value"],
                }
                if violation_type_normalized:
                    violation_context["report_violation_type_normalized"] = violation_type_normalized
                violation_context.update(report_metadata)
                entry = {
                    "id": f"repair_{qid}_{fix_event['repair_revision_id']}",
                    "qid": qid,
                    "property": pid,
                    "track": "A_BOX",
                    "information_type": "TBD",
                    "violation_context": violation_context,
                    "repair_target": {
                        "kind": "A_BOX",
                        "action": fix_event["action"],
                        "old_value": fix_event["old_value"],
                        "new_value": fix_event["new_value"],
                        "value": fix_event["new_value"],
                        "revision_id": fix_event["repair_revision_id"],
                        "author": fix_event["author"],
                    },
                    "persistence_check": {
                        "status": "passed",
                        "current_value_2025": normalized_current_values,
                    },
                }
                # Re-run constraint diffing without heavy payloads to flag ambiguous cases.
                cheap_tbox_event, ambiguous_history_meta = find_tbox_reform_revision(
                    pid,
                    start_time=start_time,
                    end_time=end_time,
                    include_snapshots=False,
                    scan_from_end=True,
                    max_revisions=25,
                )
                if cheap_tbox_event:
                    entry["ambiguous"] = True
                    entry["ambiguous_reasons"] = ["A_BOX_CHANGED", "T_BOX_CHANGED"]
                    summary["ambiguous_both_changed"] += 1
                entry_id = entry["id"]
                if entry_id in seen_ids:
                    progress.write(f"    [!] Duplicate Stage-2 id detected ({entry_id}). Skipping.")
                    summary["duplicates_skipped"] += 1
                    stats_logger.log(
                        {
                            **record_base,
                            "result": "duplicate_stage2_id",
                            "track": "A_BOX",
                            "duplicate_id": entry_id,
                        }
                    )
                else:
                    seen_ids.add(entry_id)
                    dataset.append(enrich_repair_entry(entry, label_resolver))
                    entity_history_scanned = history_scanned(history_meta)
                    property_history_scanned = history_scanned(ambiguous_history_meta)
                    stats_payload = {
                        **record_base,
                        "result": "repair_found",
                        "track": "A_BOX",
                        "history": history_meta,
                        "history_entity": history_meta,
                        "repair_revision_id": fix_event["repair_revision_id"],
                        "action": fix_event["action"],
                    }
                    if entry.get("ambiguous"):
                        stats_payload["ambiguous"] = True
                    if ambiguous_history_meta:
                        stats_payload["history_property"] = ambiguous_history_meta
                    stats_payload["entity_history_scanned"] = entity_history_scanned
                    stats_payload["property_history_scanned"] = property_history_scanned
                    stats_logger.log(stats_payload)
            else:
                # Reformer (T-box) path: fall back to constraint evolution when the A-box stayed untouched.
                tbox_event, tbox_history_meta = find_tbox_reform_revision(
                    pid,
                    start_time=start_time,
                    end_time=end_time,
                )
                if tbox_event:
                    current_values_live = get_current_state(qid, pid)
                    if current_values_live is None and STRICT_PERSISTENCE:
                        progress.write("    [x] Dropped: Persistence check failed (Entity/Prop missing).")
                        summary["persistence_failed"] += 1
                        stats_logger.log(
                            {
                                **record_base,
                                "result": "persistence_failed",
                                "reason": "missing_current_value",
                                "track": "T_BOX",
                                "property_revision_id": tbox_event["property_revision_id"],
                            }
                        )
                        progress.update(1)
                        continue
                    normalized_current_values = current_values_live if current_values_live is not None else []
                    delta = tbox_event["constraint_delta"]
                    progress.write(
                        f"    [+] FOUND T-BOX REFORM! signature {delta['hash_before']} -> {delta['hash_after']}"
                    )
                    summary["repairs_found"] += 1
                    summary["repairs_found_t_box"] += 1
                    violation_context = {
                        "report_violation_type": violation_type,
                        "value": None,
                        "value_current_2025": normalized_current_values,
                    }
                    if violation_type_normalized:
                        violation_context["report_violation_type_normalized"] = violation_type_normalized
                    violation_context.update(report_metadata)
                    entry = {
                        # Include qid so every T-box ID stays globally unique across focus nodes.
                        "id": f"reform_{qid}_{pid}_{tbox_event['property_revision_id']}",
                        "qid": qid,
                        "property": pid,
                        "track": "T_BOX",
                        "information_type": "TBD",
                        "violation_context": violation_context,
                        "repair_target": {
                            "kind": "T_BOX",
                            "property_revision_id": tbox_event["property_revision_id"],
                            "property_revision_prev": tbox_event["property_revision_prev"],
                            "author": tbox_event["author"],
                            "constraint_delta": delta,
                        },
                        "persistence_check": {
                            "status": "passed",
                            "current_value_2025": normalized_current_values,
                        },
                    }
                    entry_id = entry["id"]
                    if entry_id in seen_ids:
                        progress.write(f"    [!] Duplicate Stage-2 id detected ({entry_id}). Skipping.")
                        summary["duplicates_skipped"] += 1
                        stats_logger.log(
                            {
                                **record_base,
                                "result": "duplicate_stage2_id",
                                "track": "T_BOX",
                                "duplicate_id": entry_id,
                            }
                        )
                    else:
                        seen_ids.add(entry_id)
                        dataset.append(enrich_repair_entry(entry, label_resolver))
                        entity_history_scanned = history_scanned(history_meta)
                        property_history_scanned = history_scanned(tbox_history_meta)
                        stats_logger.log(
                            {
                                **record_base,
                                "result": "repair_found",
                                "track": "T_BOX",
                                "history": history_meta,
                                "history_entity": history_meta,
                                "history_property": tbox_history_meta,
                                "property_revision_id": tbox_event["property_revision_id"],
                                "constraint_hash_before": delta["hash_before"],
                                "constraint_hash_after": delta["hash_after"],
                                "entity_history_scanned": entity_history_scanned,
                                "property_history_scanned": property_history_scanned,
                            }
                        )
                else:
                    progress.write("    [-] No clean diff found (A-box or T-box).")
                    entity_history_scanned = history_scanned(history_meta)
                    property_history_scanned = history_scanned(tbox_history_meta)
                    has_history = entity_history_scanned or property_history_scanned
                    if has_history:
                        summary["no_diff"] += 1
                    else:
                        summary["no_history"] += 1
                    stats_logger.log(
                        {
                            **record_base,
                            "result": "no_diff" if has_history else "no_history",
                            "history": history_meta,
                            "history_entity": history_meta,
                            "history_property": tbox_history_meta,
                            "entity_history_scanned": entity_history_scanned,
                            "property_history_scanned": property_history_scanned,
                        }
                    )

            record_history_stats(history_meta)
            record_history_stats(tbox_history_meta)
            record_history_stats(ambiguous_history_meta)

            if i % 10 == 0:
                with open(WIKIDATA_REPAIRS, "w") as out:
                    json.dump(dataset, out, indent=2)
            progress.update(1)
        progress.close()
        if summary:
            with open(WIKIDATA_REPAIRS, "w", encoding="utf-8") as out:
                json.dump(dataset, out, indent=2)

    if dataset:
        builder = WorldStateBuilder(LATEST_DUMP_PATH)
        repair_ids, total_entries = extract_repair_ids(dataset)
        ensure_all_entries_have_ids(total_entries, repair_ids)
        expected_ids = ensure_unique_ids(repair_ids)
        world_state_map = {}
        for entry_id, context in builder.build(dataset):
            if entry_id in world_state_map:
                raise ValueError(f"Duplicate world_state id detected during build: {entry_id}")
            validate_world_state_entry(entry_id, context, expected_ids)
            world_state_map[entry_id] = context
        missing_ids = expected_ids - set(world_state_map.keys())
        if missing_ids:
            print(
                f"[!] Context builder missing {len(missing_ids)} entries "
                f"(likely focus entities absent in dump). Dropping them from Stage-2."
            )
            if summary is not None:
                summary["world_state_missing"] = summary.get("world_state_missing", 0) + len(missing_ids)
            dataset = [entry for entry in dataset if entry.get("id") not in missing_ids]
            with open(WIKIDATA_REPAIRS, "w", encoding="utf-8") as out:
                json.dump(dataset, out, indent=2)
            repair_ids, total_entries = extract_repair_ids(dataset)
            ensure_all_entries_have_ids(total_entries, repair_ids)
            expected_ids = ensure_unique_ids(repair_ids)
        validate_world_state_document(world_state_map, expected_ids)
        with open(WORLD_STATE_FILE, "w", encoding="utf-8") as world_file:
            json.dump(world_state_map, world_file, indent=2)

    if summary:
        with open(SUMMARY_FILE, "w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2)

    print(f"\n[+] Extraction Complete. Saved {len(dataset)} verified repairs.")


def parse_args():
    parser = argparse.ArgumentParser(description="WikidataRepairEval Phase-1 pipeline helper.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the generated world state file against Stage-2 repairs and exit.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Limit the number of repair candidates processed (debugging helper).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.validate_only:
        validate_world_state_file(WORLD_STATE_FILE, WIKIDATA_REPAIRS)
        return
    process_pipeline(max_candidates=args.max_candidates)


if __name__ == "__main__":
    main()
