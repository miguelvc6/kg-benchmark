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
    "P569",  # Date of Birth
    "P570",  # Date of Death
    "P21",  # Sex or Gender
]
REPORT_HISTORY_DEPTH = 20  # Revision pairs scanned per report page
QID_PATTERN = re.compile(r"\[\[(Q\d+)\]\]")
SITE = None

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
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
    current_section = "Unknown"

    # Matches headers like "== Format ==" or "=== Single value ==="
    header_pattern = re.compile(r"^={2,}\s*([^=]+?)\s*={2,}\s*$")

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Detect section header
        header_match = header_pattern.match(line)
        if header_match:
            current_section = header_match.group(1).strip()
            continue

        # Extract QIDs in this line
        qids = QID_PATTERN.findall(line)
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
                },
                "after": {
                    "hash": delta.get("hash_after"),
                    "signature": delta.get("signature_after"),
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
    serialized = json.dumps(normalized, ensure_ascii=True, sort_keys=True)
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
        "signature_before": previous_signature["signature"],
        "signature_after": current_signature["signature"],
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


def process_pipeline(max_candidates=None):
    """Main entry point: reads candidates, finds repairs, and builds context."""
    input_file = REPAIR_CANDIDATES_FILE
    candidates = ensure_repair_candidates_file(input_file)
    if not candidates:
        print(f"[!] Unable to proceed without {input_file}.")
        return

    dataset = load_cached_repairs(WIKIDATA_REPAIRS)

    summary = None
    if dataset is not None:
        print(
            "[*] Using cached repairs file for Stage 3. Delete data/02_wikidata_repairs.json to force recompute Stage 2."
        )
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
                entry = {
                    "id": f"repair_{qid}_{fix_event['repair_revision_id']}",
                    "qid": qid,
                    "property": pid,
                    "track": "A_BOX",
                    "type": violation_type or "TBD",
                    "violation_context": {
                        "report_violation_type": violation_type,
                        "value": fix_event["old_value"],
                    },
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
                entry["violation_context"].update(report_metadata)
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
                dataset.append(entry)
                entity_history_scanned = history_scanned(history_meta)
                property_history_scanned = history_scanned(tbox_history_meta)
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
                    entry = {
                        # Include qid so every T-box ID stays globally unique across focus nodes.
                        "id": f"reform_{qid}_{pid}_{tbox_event['property_revision_id']}",
                        "qid": qid,
                        "property": pid,
                        "track": "T_BOX",
                        "type": violation_type or "TBD",
                        "violation_context": {
                            "report_violation_type": violation_type,
                            "value": None,
                            "value_current_2025": normalized_current_values,
                        },
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
                    entry["violation_context"].update(report_metadata)
                    dataset.append(entry)
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
        produced_any = False
        with open(WORLD_STATE_FILE, "w", encoding="utf-8") as world_file:
            world_file.write("{")
            first = True
            buffer_entries = []
            emitted_ids = set()

            def flush_world_state_buffer():
                nonlocal first, produced_any, buffer_entries
                if not buffer_entries:
                    return
                for buffered_id, buffered_context in buffer_entries:
                    produced_any = True
                    json_context = json.dumps(buffered_context, indent=2).replace("\n", "\n  ")
                    if first:
                        world_file.write("\n")
                        first = False
                    else:
                        world_file.write(",\n")
                    world_file.write(f'  "{buffered_id}": {json_context}')
                buffer_entries = []

            for entry_id, context in builder.build(dataset):
                if entry_id in emitted_ids:
                    print(f"[!] Duplicate world_state id {entry_id} detected. Skipping to avoid overwriting context.")
                    continue
                emitted_ids.add(entry_id)
                buffer_entries.append((entry_id, context))
                if len(buffer_entries) >= 10:
                    flush_world_state_buffer()
            flush_world_state_buffer()
            world_file.write("\n}\n" if produced_any else "}\n")

    if summary:
        with open(SUMMARY_FILE, "w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2)

    print(f"\n[+] Extraction Complete. Saved {len(dataset)} verified repairs.")


if __name__ == "__main__":
    process_pipeline()
