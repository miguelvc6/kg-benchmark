import argparse
import json
import logging
import random
import time
from datetime import UTC, datetime

from tqdm import tqdm

from lib.caching import (
    REVISION_HISTORY_CACHE,
    SNAPSHOT_FETCHER,
    LabelResolver,
    extract_user,
    fetch_revision_history,
    get_claims_from_snapshot,
    get_current_state,
    get_entity_snapshot,
)
from lib.config import (
    LATEST_DUMP_PATH,
    MAX_HISTORY_PAGES,
    REPAIR_CANDIDATES_FILE,
    RESUME_CHECKPOINT_EVERY,
    RESUME_DEFAULT_CHECKPOINT,
    REVISION_LOOKBACK_DAYS,
    RUN_ID,
    SNAPSHOT_PREFETCH,
    STAGE2_LOG_EVERY,
    STATS_FILE,
    STATS_FLUSH_EVERY,
    STRICT_PERSISTENCE,
    SUMMARY_FILE,
    TARGET_PROPERTIES,
    WIKIDATA_REPAIRS,
    WIKIDATA_REPAIRS_JSONL,
    WORLD_STATE_FILE,
)
from lib.mining import (
    build_report_provenance,
    deduplicate_candidates,
    ensure_repair_candidates_file,
    normalize_report_violation_type,
)
from lib.popularity import attach_entity_popularity, ensure_entity_popularity
from lib.utils import (
    _build_constraint_delta,
    append_jsonl_record,
    build_candidate_key,
    classify_action,
    compile_jsonl_to_json,
    compute_revision_window,
    enrich_repair_entries,
    enrich_repair_entry,
    load_cached_repairs,
    load_jsonl_ids,
    load_resume_checkpoint,
    load_resume_stats,
    parse_iso8601,
    signature_p2302,
    summarize_claims,
    write_resume_checkpoint,
)
from lib.world_state import (
    WorldStateBuilder,
    ensure_all_entries_have_ids,
    ensure_unique_ids,
    extract_repair_ids,
    validate_world_state_document,
    validate_world_state_entry,
    validate_world_state_file,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class StatsLogger:
    """Append-only JSONL logger for per-candidate fetch diagnostics."""

    def __init__(self, stats_path):
        """Initialize logger with target file and shared run_id."""
        self.stats_path = stats_path
        self.run_id = RUN_ID
        self.buffer = []

    def log(self, record):
        """Write a single JSON object line enriched with the run identifier."""
        enriched = {"run_id": self.run_id}
        enriched.update(record)
        self.buffer.append(json.dumps(enriched, ensure_ascii=True))
        if len(self.buffer) >= STATS_FLUSH_EVERY:
            self.flush()

    def flush(self):
        """Flush any buffered JSONL lines to disk."""
        if not self.buffer:
            return
        with open(self.stats_path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(self.buffer))
            fh.write("\n")
        self.buffer.clear()


def find_repair_revision(qid, property_id, start_time, end_time):
    """Walk revision history to locate the first diff that changes the property."""
    revisions, history_meta = fetch_revision_history(qid, start_time, end_time)
    if not revisions:
        return None, history_meta

    ordered_revisions = list(revisions)
    ordered_revisions.sort(key=lambda rev: rev.get("_rev_dt") or parse_iso8601(rev.get("timestamp")))
    newest_first = list(reversed(ordered_revisions))
    revision_items = []
    for rev in newest_first:
        revision_id = rev.get("id") or rev.get("revid")
        if revision_id:
            revision_items.append((rev, revision_id))

    revision_ids = [rev_id for _, rev_id in revision_items]
    if revision_ids:
        SNAPSHOT_FETCHER.prefetch(qid, revision_ids[:SNAPSHOT_PREFETCH], max_in_flight=SNAPSHOT_PREFETCH)

    newer_signature = None
    newer_snapshot = None
    newer_revision = None

    for idx, (rev, revision_id) in enumerate(revision_items):
        if revision_ids and idx + 1 < len(revision_ids):
            next_slice = revision_ids[idx + 1 : idx + 1 + SNAPSHOT_PREFETCH]
            SNAPSHOT_FETCHER.prefetch(qid, next_slice, max_in_flight=SNAPSHOT_PREFETCH)

        snapshot = get_entity_snapshot(qid, revision_id)
        current_claims = get_claims_from_snapshot(snapshot, property_id)
        current_signature, current_snapshot = summarize_claims(current_claims)

        if newer_signature is None:
            newer_signature = current_signature
            newer_snapshot = current_snapshot
            newer_revision = rev
            continue

        if current_signature != newer_signature:
            repair_revision_id = newer_revision.get("id") or newer_revision.get("revid")
            return (
                {
                    "repair_revision_id": repair_revision_id,
                    "timestamp": newer_revision.get("timestamp"),
                    "action": classify_action(current_signature, newer_signature),
                    "old_value": current_snapshot,
                    "new_value": newer_snapshot,
                    "author": extract_user(newer_revision),
                },
                history_meta,
            )

        newer_signature = current_signature
        newer_snapshot = current_snapshot
        newer_revision = rev

    return None, history_meta


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
        reversed_revisions = list(reversed(ordered_revisions))
        revision_items = []
        for rev in reversed_revisions:
            revision_id = rev.get("id") or rev.get("revid")
            if revision_id:
                revision_items.append((rev, revision_id))
        revision_ids = [rev_id for _, rev_id in revision_items]
        if revision_ids:
            SNAPSHOT_FETCHER.prefetch(property_id, revision_ids[:SNAPSHOT_PREFETCH], max_in_flight=SNAPSHOT_PREFETCH)

        for idx, (rev, revision_id) in enumerate(revision_items):
            if max_revisions is not None and scanned >= max_revisions:
                break
            if revision_ids and idx + 1 < len(revision_ids):
                next_slice = revision_ids[idx + 1 : idx + 1 + SNAPSHOT_PREFETCH]
                SNAPSHOT_FETCHER.prefetch(property_id, next_slice, max_in_flight=SNAPSHOT_PREFETCH)
            snapshot = get_entity_snapshot(property_id, revision_id)
            constraint_claims = get_claims_from_snapshot(snapshot, "P2302")
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
        snapshot = get_entity_snapshot(property_id, revision_id)
        constraint_claims = get_claims_from_snapshot(snapshot, "P2302")
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


def process_pipeline(
    max_candidates=None,
    resume_stats=None,
    resume_checkpoint=None,
):
    """Main entry point: reads candidates, finds repairs, and builds context."""
    # Step 1: Load candidates and de-duplicate
    input_file = REPAIR_CANDIDATES_FILE
    candidates = ensure_repair_candidates_file(input_file)
    if not candidates:
        logger.warning("[!] Unable to proceed without %s.", input_file)
        return
    raw_candidate_count = len(candidates)
    candidates, dedup_stats = deduplicate_candidates(candidates)
    random.seed(42)
    random.shuffle(candidates)
    if dedup_stats.get("duplicates_skipped"):
        logger.info(
            "[*] Deduplicated candidates: %s skipped, %s merged violation types.",
            dedup_stats["duplicates_skipped"],
            dedup_stats["violation_type_merges"],
        )

    label_resolver = LabelResolver()
    dataset = load_cached_repairs(WIKIDATA_REPAIRS)

    summary = None
    if dataset is not None:
        logger.info(
            "[*] Using cached repairs file for Stage 3. Delete data/02_wikidata_repairs.json to force recompute Stage 2."
        )
        enrich_repair_entries(dataset, label_resolver)
        with open(WIKIDATA_REPAIRS, "w", encoding="utf-8") as out:
            json.dump(dataset, out, indent=2)
    else:
        # Step 2: Resume planning and Stage-2 analysis loop
        stats_logger = StatsLogger(STATS_FILE)
        resume_info = {
            "enabled": False,
            "stats_path": None,
            "checkpoint_path": None,
            "checkpoint_output_path": None,
            "start_index": 0,
            "skip_keys": set(),
            "resume_stats_lines": 0,
            "resume_skipped": 0,
        }
        resume_start_index = 0
        resume_checkpoint_payload = load_resume_checkpoint(resume_checkpoint)
        if resume_checkpoint_payload and isinstance(resume_checkpoint_payload.get("last_index"), int):
            resume_start_index = resume_checkpoint_payload["last_index"] + 1
            resume_info["enabled"] = True
            resume_info["checkpoint_path"] = str(resume_checkpoint)
            logger.info(
                "[*] Resume checkpoint loaded: last_index=%s processed=%s",
                resume_checkpoint_payload["last_index"],
                resume_checkpoint_payload.get("processed_count", 0),
            )

        resume_stats_data = load_resume_stats(resume_stats, include_coarse=False)
        resume_info["resume_stats_lines"] = resume_stats_data["line_count"]
        if resume_stats:
            resume_info["stats_path"] = str(resume_stats)
            if resume_stats_data["has_candidate_keys"]:
                resume_info["skip_keys"] = resume_stats_data["processed_keys"]
                resume_info["enabled"] = True
                logger.info(
                    "[*] Resume stats loaded with candidate keys: %s processed entries.",
                    len(resume_info["skip_keys"]),
                )
            elif resume_stats_data["last_record"]:
                last_record = resume_stats_data["last_record"]
                last_qid = last_record.get("qid")
                last_pid = last_record.get("property")
                last_violation = last_record.get("violation_type")
                last_index = -1
                match_count = 0
                for idx, item in enumerate(candidates):
                    if (
                        item.get("qid") == last_qid
                        and item.get("property_id") == last_pid
                        and item.get("violation_type") == last_violation
                    ):
                        last_index = idx
                        match_count += 1
                if last_index >= 0:
                    resume_start_index = max(resume_start_index, last_index + 1)
                    resume_info["enabled"] = True
                    resume_info["start_index"] = resume_start_index
                    logger.info(
                        "[*] Resume stats last record matched: start_index=%s matches=%s.",
                        resume_start_index,
                        match_count,
                    )
                else:
                    logger.warning("[!] Resume stats last record did not match any candidate.")
        if resume_start_index >= len(candidates):
            logger.warning("[!] Resume start index exceeds candidate count; nothing to process.")
            return

        if resume_info["enabled"] and not resume_info["start_index"]:
            resume_info["start_index"] = resume_start_index
        resume_info["checkpoint_output_path"] = str(RESUME_DEFAULT_CHECKPOINT)
        summary = {
            "run_id": stats_logger.run_id,
            "lookback_days": REVISION_LOOKBACK_DAYS,
            "max_history_pages": MAX_HISTORY_PAGES,
            "total_candidates_raw": raw_candidate_count,
            "total_candidates": len(candidates),
            "candidate_duplicates_skipped": dedup_stats.get("duplicates_skipped", 0),
            "candidate_violation_type_merges": dedup_stats.get("violation_type_merges", 0),
            "candidate_exact_duplicates": dedup_stats.get("exact_duplicates", 0),
            "repairs_jsonl_existing": 0,
            "repairs_jsonl_written": 0,
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
            "entity_snapshot_cache_hits": 0,
            "entity_snapshot_cache_misses": 0,
            "entity_snapshot_negative_hits": 0,
            "entity_snapshot_disk_hits": 0,
            "entity_snapshot_disk_writes": 0,
            "entity_snapshot_network_calls": 0,
            "entity_snapshot_network_errors": 0,
            "history_cache_hits": 0,
            "history_cache_misses": 0,
            "history_cache_segment_hits": 0,
            "resume_enabled": resume_info["enabled"],
            "resume_stats_path": resume_info["stats_path"],
            "resume_checkpoint_path": resume_info["checkpoint_path"],
            "resume_checkpoint_output_path": resume_info["checkpoint_output_path"],
            "resume_start_index": resume_start_index,
            "resume_stats_lines": resume_info["resume_stats_lines"],
            "resume_skipped": 0,
        }

        existing_repairs, seen_ids = load_jsonl_ids(WIKIDATA_REPAIRS_JSONL)
        summary["repairs_jsonl_existing"] = existing_repairs

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

        logger.info("[*] Loaded %s candidates. Using REST history.", len(candidates))
        stage2_start = time.time()
        last_log_time = stage2_start
        last_log_processed = 0

        def log_stage2_progress(processed_count):
            nonlocal last_log_time, last_log_processed
            if processed_count <= 0 or processed_count % STAGE2_LOG_EVERY != 0:
                return
            now = time.time()
            interval = processed_count - last_log_processed
            elapsed = now - last_log_time
            avg_seconds = (elapsed / interval) if interval else 0.0
            snapshot_stats = SNAPSHOT_FETCHER.stats
            snapshot_hits = snapshot_stats.get("cache_hits", 0)
            snapshot_misses = snapshot_stats.get("cache_misses", 0)
            snapshot_den = snapshot_hits + snapshot_misses
            snapshot_hit_rate = snapshot_hits / snapshot_den if snapshot_den else 0.0
            label_hits = label_resolver.stats.get("db_hits", 0)
            label_misses = label_resolver.stats.get("db_misses", 0)
            label_den = label_hits + label_misses
            label_hit_rate = label_hits / label_den if label_den else 0.0
            status_counts = snapshot_stats.get("http_status_counts", {})
            inflight = SNAPSHOT_FETCHER.inflight_size()
            progress.write(
                "[*] Stage-2 perf: "
                f"{avg_seconds:.3f}s/entity (rolling), "
                f"snapshot hit {snapshot_hit_rate:.2%}, "
                f"label hit {label_hit_rate:.2%}, "
                f"inflight {inflight}, "
                f"HTTP 200={status_counts.get(200, 0)} "
                f"400={status_counts.get(400, 0)} "
                f"404={status_counts.get(404, 0)} "
                f"429={status_counts.get(429, 0)} "
                f"other={status_counts.get('other', 0)}"
            )
            last_log_time = now
            last_log_processed = processed_count

        remaining_candidates = candidates[resume_start_index:]
        total_to_process = len(remaining_candidates)
        if max_candidates is not None:
            total_to_process = min(total_to_process, max_candidates)
            remaining_candidates = remaining_candidates[:total_to_process]
        try:
            with open(WIKIDATA_REPAIRS_JSONL, "a", encoding="utf-8") as repairs_file:
                progress = tqdm(total=total_to_process, desc="Processing candidates", unit="candidate")
                last_candidate_key = None
                last_index = None
                last_qid = None
                last_pid = None
                last_violation = None

                def finish_candidate():
                    progress.update(1)
                    log_stage2_progress(summary["processed"])

                for offset, item in enumerate(remaining_candidates):
                    i = resume_start_index + offset

                    def log_candidate(message):
                        if offset < 10:
                            progress.write(message)

                    qid = item["qid"]
                    pid = item["property_id"]
                    violation_type = item.get("violation_type")
                    violation_types = item.get("violation_types")
                    violation_type_normalized = normalize_report_violation_type(violation_type)
                    candidate_key = build_candidate_key(item)
                    last_candidate_key = candidate_key
                    last_index = i
                    last_qid = qid
                    last_pid = pid
                    last_violation = violation_type

                    if resume_info["skip_keys"] and candidate_key in resume_info["skip_keys"]:
                        summary["resume_skipped"] += 1
                        finish_candidate()
                        continue

                    if not qid.startswith("Q"):
                        finish_candidate()
                        continue

                    if TARGET_PROPERTIES and pid not in TARGET_PROPERTIES:
                        finish_candidate()
                        continue

                    log_candidate(f"[{i + 1}/{total_to_process}] Analyzing {qid} ({pid})...")
                    summary["processed"] += 1

                    record_base = {
                        "qid": qid,
                        "property": pid,
                        "violation_type": violation_type,
                        "candidate_key": candidate_key,
                        "candidate_index": i,
                        "fix_date": item.get("fix_date"),
                        "report_revision_old": item.get("report_revision_old"),
                        "report_revision_new": item.get("report_revision_new"),
                    }
                    report_metadata = build_report_provenance(item, pid)

                    report_date = item["fix_date"]
                    start_time, end_time = compute_revision_window(report_date)
                    if not end_time:
                        log_candidate("    [x] Dropped: Could not parse fix_date.")
                        summary["bad_fix_date"] += 1
                        stats_logger.log(
                            {
                                **record_base,
                                "result": "bad_fix_date",
                                "report_date": report_date,
                            }
                        )
                        finish_candidate()
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
                        log_candidate(
                            f"    [+] FOUND A-BOX REPAIR! {fix_event['old_value']} -> {fix_event['new_value']}"
                        )
                        summary["repairs_found"] += 1
                        summary["repairs_found_a_box"] += 1
                        current_values_live = get_current_state(qid, pid)
                        if current_values_live is None and STRICT_PERSISTENCE and fix_event["action"] != "DELETE":
                            log_candidate("    [x] Dropped: Persistence check failed (Entity/Prop missing).")
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
                            finish_candidate()
                            continue
                        normalized_current_values = current_values_live if current_values_live is not None else []
                        violation_context = {
                            "report_violation_type": violation_type,
                            "value": fix_event["old_value"],
                        }
                        if violation_types:
                            violation_context["report_violation_types"] = violation_types
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
                                "current_value_2026": normalized_current_values,
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
                            log_candidate(f"    [!] Duplicate Stage-2 id detected ({entry_id}). Skipping.")
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
                            entry = enrich_repair_entry(entry, label_resolver)
                            append_jsonl_record(repairs_file, entry)
                            summary["repairs_jsonl_written"] += 1
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
                                log_candidate("    [x] Dropped: Persistence check failed (Entity/Prop missing).")
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
                                finish_candidate()
                                continue
                            normalized_current_values = current_values_live if current_values_live is not None else []
                            delta = tbox_event["constraint_delta"]
                            log_candidate(
                                f"    [+] FOUND T-BOX REFORM! signature {delta['hash_before']} -> {delta['hash_after']}"
                            )
                            summary["repairs_found"] += 1
                            summary["repairs_found_t_box"] += 1
                            violation_context = {
                                "report_violation_type": violation_type,
                                "value": None,
                                "value_current_2026": normalized_current_values,
                            }
                            if violation_types:
                                violation_context["report_violation_types"] = violation_types
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
                                    "current_value_2026": normalized_current_values,
                                },
                            }
                            entry_id = entry["id"]
                            if entry_id in seen_ids:
                                log_candidate(f"    [!] Duplicate Stage-2 id detected ({entry_id}). Skipping.")
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
                                entry = enrich_repair_entry(entry, label_resolver)
                                append_jsonl_record(repairs_file, entry)
                                summary["repairs_jsonl_written"] += 1
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
                            log_candidate("    [-] No clean diff found (A-box or T-box).")
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

                    finish_candidate()

                    if summary["processed"] % RESUME_CHECKPOINT_EVERY == 0:
                        checkpoint_payload = {
                            "run_id": stats_logger.run_id,
                            "processed_count": summary["processed"],
                            "last_index": last_index,
                            "last_candidate_key": last_candidate_key,
                            "last_qid": last_qid,
                            "last_property": last_pid,
                            "last_violation_type": last_violation,
                            "timestamp_utc": datetime.now(UTC).isoformat(),
                        }
                        write_resume_checkpoint(RESUME_DEFAULT_CHECKPOINT, checkpoint_payload)
                progress.close()
                if last_index is not None:
                    checkpoint_payload = {
                        "run_id": stats_logger.run_id,
                        "processed_count": summary["processed"],
                        "last_index": last_index,
                        "last_candidate_key": last_candidate_key,
                        "last_qid": last_qid,
                        "last_property": last_pid,
                        "last_violation_type": last_violation,
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                        "completed": True,
                    }
                    write_resume_checkpoint(RESUME_DEFAULT_CHECKPOINT, checkpoint_payload)
        finally:
            stats_logger.flush()
        if summary:
            snapshot_stats = SNAPSHOT_FETCHER.stats
            summary["entity_snapshot_cache_hits"] = snapshot_stats.get("cache_hits", 0)
            summary["entity_snapshot_cache_misses"] = snapshot_stats.get("cache_misses", 0)
            summary["entity_snapshot_negative_hits"] = snapshot_stats.get("negative_hits", 0)
            summary["entity_snapshot_disk_hits"] = snapshot_stats.get("disk_hits", 0)
            summary["entity_snapshot_disk_writes"] = snapshot_stats.get("disk_writes", 0)
            summary["entity_snapshot_network_calls"] = snapshot_stats.get("network_calls", 0)
            summary["entity_snapshot_network_errors"] = snapshot_stats.get("network_errors", 0)
            if REVISION_HISTORY_CACHE:
                summary["history_cache_hits"] = REVISION_HISTORY_CACHE.hits
                summary["history_cache_misses"] = REVISION_HISTORY_CACHE.misses
                summary["history_cache_segment_hits"] = REVISION_HISTORY_CACHE.segment_hits
            compile_jsonl_to_json(WIKIDATA_REPAIRS_JSONL, WIKIDATA_REPAIRS)
            dataset = load_cached_repairs(WIKIDATA_REPAIRS) or []
        else:
            dataset = []

    if dataset:
        # Step 3: Enrich dataset and build world state context
        popularity_map = ensure_entity_popularity(dataset)
        attach_entity_popularity(dataset, popularity_map)
        with open(WIKIDATA_REPAIRS, "w", encoding="utf-8") as out:
            json.dump(dataset, out, indent=2)
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
            logger.warning(
                "[!] Context builder missing %s entries (likely focus entities absent in dump). Dropping them from Stage-2.",
                len(missing_ids),
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

        logger.info("[+] Extraction Complete. Saved %s verified repairs.", len(dataset))


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
    parser.add_argument(
        "--resume-stats",
        type=str,
        default=None,
        help="Path to a prior fetcher_stats_*.jsonl to resume from.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to a resume checkpoint JSON produced by a previous run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.validate_only:
        validate_world_state_file(WORLD_STATE_FILE, WIKIDATA_REPAIRS)
        return
    process_pipeline(
        max_candidates=args.max_candidates,
        resume_stats=args.resume_stats,
        resume_checkpoint=args.resume_checkpoint,
    )


if __name__ == "__main__":
    main()
