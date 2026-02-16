import copy
import gzip
import json
import logging
import sys
import time
from pathlib import Path

import ijson
from tqdm import tqdm

from .caching import LabelResolver
from . import config
from .utils import format_datavalue, iter_repairs, pick_description, pick_label

logger = logging.getLogger(__name__)


def _format_elapsed(seconds):
    """Return compact HH:MM:SS elapsed display."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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
    for required in config.REQUIRED_WORLD_STATE_KEYS:
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
    repair_ids = []
    total_entries = 0
    for entry in iter_repairs(repairs_path):
        if not isinstance(entry, dict):
            continue
        total_entries += 1
        entry_id = entry.get("id")
        if entry_id:
            repair_ids.append(entry_id)
    if total_entries == 0:
        raise RuntimeError(f"Stage-2 repairs file missing or empty at {repairs_path}.")
    ensure_all_entries_have_ids(total_entries, repair_ids)
    expected_ids = ensure_unique_ids(repair_ids)
    path = Path(world_state_path)
    if not path.exists():
        raise RuntimeError(f"World state file not found at {world_state_path}.")
    heartbeat_every_seconds = max(1, int(config.PROGRESS_HEARTBEAT_SECONDS))
    validate_start = time.monotonic()
    last_heartbeat = validate_start
    seen_ids = set()
    with open(path, "rb") as fh:
        for idx, (entry_id, entry) in enumerate(ijson.kvitems(fh, ""), start=1):
            _ensure(entry_id not in seen_ids, f"Duplicate world state key encountered: {entry_id}")
            validate_world_state_entry(entry_id, entry, expected_ids)
            seen_ids.add(entry_id)
            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_every_seconds:
                elapsed = now - validate_start
                rate = idx / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "[*] World state validate heartbeat: checked %s entries in %s (%.2f entries/s).",
                    f"{idx:,}",
                    _format_elapsed(elapsed),
                    rate,
                )
                last_heartbeat = now
    missing = expected_ids - seen_ids
    unexpected = seen_ids - expected_ids
    _ensure(
        not missing,
        f"World state missing {len(missing)} ids from Stage-2 dataset: {sorted(list(missing))[:5]} ...",
    )
    _ensure(
        not unexpected,
        f"World state has unexpected ids not found in Stage-2 dataset: {sorted(list(unexpected))[:5]} ...",
    )
    print(f"[+] World state validation passed for {len(seen_ids)} entries.")


class WorldStateBuilder:
    """Streams the 2026 dump to attach ego/neighbor context to each repair."""

    def __init__(self, dump_path):
        """Record dump location and flag whether it is available."""
        self.dump_path = Path(dump_path)
        self.has_dump = self.dump_path.exists()
        self.label_resolver = LabelResolver()

    def build(self, entries):
        """Yield world_state payloads for the provided repair entries."""
        if not entries:
            return
        if not self.has_dump:
            logger.warning("[!] Context Builder skipped: dump not found at %s", self.dump_path)
            return

        # 1. Identify Initial Targets (Focus + Property)
        focus_ids = {entry["qid"] for entry in entries}
        property_ids = {entry["property"] for entry in entries}
        target_ids = focus_ids | property_ids

        logger.info("[*] Context Builder: Single-pass stream for %s primary entities...", len(target_ids))

        # 2. Single Pass Scan
        # We only load the Focus and Property entities from the dump.
        retrieved_entities = self._load_entities_from_dump(target_ids)

        focus_entities = {eid: retrieved_entities[eid] for eid in focus_ids if eid in retrieved_entities}
        property_entities = {pid: retrieved_entities[pid] for pid in property_ids if pid in retrieved_entities}

        missing_focus = focus_ids - set(focus_entities)
        if missing_focus:
            logger.warning("[!] Warning: %s focus entities not found in dump.", len(missing_focus))

        # 3. Identify Neighbors from the loaded data
        neighbor_ids = self._collect_neighbor_targets(focus_entities.values())
        constraint_target_ids = self._collect_constraint_targets(property_entities)

        # Filter out neighbors we already happened to load (if they were focus nodes too)
        preloaded_ids = set(retrieved_entities.keys())
        api_label_targets = (neighbor_ids | constraint_target_ids) - preloaded_ids

        logger.info(
            "[*] Context Builder: Fetching %s neighbor/constraint labels via API (avoiding 2nd dump scan)...",
            len(api_label_targets),
        )

        # 4. API Fallback for Neighbors + constraint references (Much faster than 2nd scan)
        # We fetch these just for labels/descriptions
        self._prefetch_labels_via_api(api_label_targets)

        # Keep only dump-sourced entities in memory; labels for API-only ids are
        # looked up lazily via LabelResolver (SQLite-backed).
        full_entity_map = retrieved_entities

        total_entries = len(entries)
        heartbeat_every_seconds = max(1, int(config.PROGRESS_HEARTBEAT_SECONDS))
        start_time = time.monotonic()
        last_heartbeat = start_time
        for idx, entry in enumerate(
            tqdm(
                entries,
                desc="Building world states",
                unit="entry",
                disable=not sys.stderr.isatty(),
            ),
            start=1,
        ):
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
            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_every_seconds:
                elapsed = now - start_time
                rate = idx / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "[*] Context Builder heartbeat: built %s/%s world states in %s (%.2f entries/s).",
                    idx,
                    total_entries,
                    _format_elapsed(elapsed),
                    rate,
                )
                last_heartbeat = now

    def _load_entities_from_dump(self, target_ids):
        """Single-pass stream over the dump extracting matching entity blobs."""
        found = {}
        if not target_ids or not self.has_dump:
            return found
        target_ids = set(target_ids)

        heartbeat_every_seconds = max(1, int(config.PROGRESS_HEARTBEAT_SECONDS))
        total_hint = getattr(config, "DUMP_SCAN_TOTAL_ENTITIES", None)
        scan_start = time.monotonic()
        last_heartbeat = scan_start
        scanned = 0
        try:
            with gzip.open(self.dump_path, "rb") as fh:
                stream = ijson.items(fh, "item")
                for scanned, entity in enumerate(
                    tqdm(
                        stream,
                        desc="Scanning dump for context",
                        unit=" entity",
                        miniters=10000,
                        total=total_hint,
                        disable=not sys.stderr.isatty(),
                    ),
                    start=1,
                ):
                    eid = entity.get("id")
                    if eid in target_ids:
                        found[eid] = entity
                        # Stop early if we found everything
                        if len(found) == len(target_ids):
                            tqdm.write("    [+] Found all targets. Stopping stream early.")
                            break
                    now = time.monotonic()
                    if now - last_heartbeat >= heartbeat_every_seconds:
                        elapsed = now - scan_start
                        rate = scanned / elapsed if elapsed > 0 else 0.0
                        logger.info(
                            "[*] Dump scan heartbeat: scanned %s entities, found %s/%s targets in %s (%.2f entities/s).",
                            f"{scanned:,}",
                            len(found),
                            len(target_ids),
                            _format_elapsed(elapsed),
                            rate,
                        )
                        last_heartbeat = now
        except Exception as exc:
            logger.warning(
                "[!] Dump stream error after scanning %s entities and finding %s/%s targets: %s",
                f"{scanned:,}",
                len(found),
                len(target_ids),
                exc,
            )
        return found

    def _collect_neighbor_targets(self, entities):
        """Gather unique QIDs reachable via outbound statements."""
        neighbors = set()
        for entity in tqdm(entities, desc="Collecting neighbors", unit="focus", disable=not sys.stderr.isatty()):
            edges = 0
            claims = entity.get("claims", {})
            for pid, statements in claims.items():
                for claim in statements:
                    if edges >= config.MAX_NEIGHBOR_EDGES:
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
                if edges >= config.MAX_NEIGHBOR_EDGES:
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

    def _prefetch_labels_via_api(self, ids):
        """Resolve missing labels/descriptions in batches and persist to label cache."""
        id_list = list(ids)
        if not id_list:
            return
        total_batches = (len(id_list) + 49) // 50
        heartbeat_every_seconds = max(1, int(config.PROGRESS_HEARTBEAT_SECONDS))
        fetch_start = time.monotonic()
        last_heartbeat = fetch_start
        resolved_count = 0
        for batch_index, start in enumerate(
            tqdm(
                range(0, len(id_list), 50),
                desc="Fetching neighbor labels",
                unit="batch",
                disable=not sys.stderr.isatty(),
            ),
            start=1,
        ):
            batch = id_list[start : start + 50]
            resolved_batch = self.label_resolver.resolve(batch)
            resolved_count += sum(
                1
                for payload in resolved_batch.values()
                if isinstance(payload, dict) and (payload.get("label_en") or payload.get("description_en"))
            )
            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_every_seconds:
                elapsed = now - fetch_start
                rate = batch_index / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "[*] Label fetch heartbeat: %s/%s batches, resolved %s ids in %s (%.2f batch/s).",
                    batch_index,
                    total_batches,
                    resolved_count,
                    _format_elapsed(elapsed),
                    rate,
                )
                last_heartbeat = now

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
        if entry.get("popularity"):
            focus_node["popularity"] = copy.deepcopy(entry["popularity"])
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
                if len(values) >= config.MAX_PROPERTY_VALUES:
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
                if edge_count >= config.MAX_NEIGHBOR_EDGES:
                    break
                snak = claim.get("mainsnak", {})
                datavalue = snak.get("datavalue")
                if not datavalue:
                    continue
                value = datavalue.get("value")
                if isinstance(value, dict) and value.get("entity-type") in {"item", "property"}:
                    target_id = value.get("id")
                    edges.append(
                        {
                            "property_id": pid,
                            "target_qid": target_id,
                            "target_label": self._lookup_label(target_id, neighbor_entities),
                            "target_description": self._lookup_description(target_id, neighbor_entities),
                        }
                    )
                    edge_count += 1
            if edge_count >= config.MAX_NEIGHBOR_EDGES:
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
            label = self._lookup_label(entity_id, entity_map)
            description = self._lookup_description(entity_id, entity_map)
            label_index[entity_id] = {
                "label": label or config.MISSING_LABEL_PLACEHOLDER,
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
        label = pick_label(entity) if entity else None
        if not label:
            cached = self.label_resolver.lookup(entity_id)
            label = cached.get("label_en") if isinstance(cached, dict) else None
        return label or config.MISSING_LABEL_PLACEHOLDER

    def _lookup_description(self, entity_id, entity_map):
        """Return the preferred description for entity_id when available."""
        if not entity_id:
            return None
        entity = entity_map.get(entity_id)
        description = pick_description(entity) if entity else None
        if description:
            return description
        cached = self.label_resolver.lookup(entity_id)
        if isinstance(cached, dict):
            return cached.get("description_en")
        return None

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
                property_label = (
                    self._lookup_label(qualifier_pid, full_entity_map) or config.MISSING_LABEL_PLACEHOLDER
                )
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
                        "label": constraint_label or (config.MISSING_LABEL_PLACEHOLDER if constraint_qid else None),
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
