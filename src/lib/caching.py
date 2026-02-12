import json
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import zstandard as zstd

from . import config
from .cache_sqlite import SQLiteLabelCache, SQLiteSnapshotCache
from .utils import (
    chunked,
    get_json,
    is_entity_or_property_id,
    parse_iso8601,
    pick_description,
    pick_label,
    summarize_claims,
)


class RateLimiter:
    """Token-bucket rate limiter for outbound snapshot fetches."""

    def __init__(self, max_qps):
        self.max_qps = max_qps or 0
        self._tokens = float(self.max_qps)
        self._last_check = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self):
        if self.max_qps <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_check
                self._last_check = now
                self._tokens = min(self.max_qps, self._tokens + elapsed * self.max_qps)
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                sleep_for = max(0.01, (1 - self._tokens) / self.max_qps)
            time.sleep(sleep_for)


class RevisionHistoryCache:
    """In-memory cache for revision history calls with optional overlap reuse."""

    def __init__(
        self,
        max_entries=config.HISTORY_CACHE_MAX_ENTRIES,
        max_segments_per_qid=config.HISTORY_CACHE_MAX_SEGMENTS_PER_QID,
    ):
        self.max_entries = max_entries
        self.max_segments_per_qid = max_segments_per_qid
        self._cache = OrderedDict()
        self._segments = {}
        self.hits = 0
        self.misses = 0
        self.segment_hits = 0

    def _key(self, qid, start_time, end_time):
        return (qid, start_time or "", end_time or "")

    def _evict_if_needed(self):
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    def _segment_covering(self, qid, start_dt, end_dt):
        segments = self._segments.get(qid) or []
        for segment in segments:
            if segment["start_dt"] <= start_dt and segment["end_dt"] >= end_dt:
                return segment
        return None

    def _slice_revisions(self, revisions, start_dt, end_dt):
        if not revisions:
            return []
        sliced = []
        for rev in revisions:
            rev_dt = rev.get("_rev_dt")
            if not rev_dt:
                rev_dt = parse_iso8601(rev.get("timestamp"))
                rev["_rev_dt"] = rev_dt
            if rev_dt is None:
                continue
            if start_dt and rev_dt < start_dt:
                continue
            if end_dt and rev_dt > end_dt:
                continue
            sliced.append(rev)
        return sliced

    def get(self, qid, start_time, end_time):
        key = self._key(qid, start_time, end_time)
        cached = self._cache.get(key)
        if cached:
            self._cache.move_to_end(key)
            self.hits += 1
            revisions, meta, cached_at = cached
            meta = dict(meta)
            meta.update({"cache_hit": True, "cache_scope": "exact", "cache_age_seconds": time.time() - cached_at})
            meta["api_calls"] = 0
            return revisions, meta

        start_dt = parse_iso8601(start_time) if start_time else None
        end_dt = parse_iso8601(end_time) if end_time else None
        if start_dt and end_dt:
            segment = self._segment_covering(qid, start_dt, end_dt)
            if segment:
                self.segment_hits += 1
                revisions = self._slice_revisions(segment["revisions"], start_dt, end_dt)
                meta = dict(segment["meta"])
                meta.update(
                    {
                        "qid": qid,
                        "start_time": start_time,
                        "end_time": end_time,
                        "revisions_scanned": len(revisions),
                        "earliest_revision": revisions[0]["timestamp"] if revisions else None,
                        "latest_revision": revisions[-1]["timestamp"] if revisions else None,
                        "cache_hit": True,
                        "cache_scope": "segment",
                        "cache_age_seconds": time.time() - segment["cached_at"],
                        "api_calls": 0,
                    }
                )
                return revisions, meta

        self.misses += 1
        return None, None

    def store(self, qid, start_time, end_time, revisions, history_meta):
        key = self._key(qid, start_time, end_time)
        self._cache[key] = (revisions, history_meta, time.time())
        self._cache.move_to_end(key)
        self._evict_if_needed()
        start_dt = parse_iso8601(start_time) if start_time else None
        end_dt = parse_iso8601(end_time) if end_time else None
        if start_dt and end_dt:
            segments = self._segments.setdefault(qid, [])
            segments.append(
                {
                    "start_dt": start_dt,
                    "end_dt": end_dt,
                    "revisions": revisions,
                    "meta": history_meta,
                    "cached_at": time.time(),
                }
            )
            if len(segments) > self.max_segments_per_qid:
                segments.pop(0)


class SnapshotFetcher:
    """Snapshot fetcher with SQLite cache, in-memory cache, and rate limiting."""

    _NEGATIVE_SENTINEL = object()

    def __init__(
        self,
        cache_db=config.ENTITY_SNAPSHOT_DB,
        enable_cache=config.ENABLE_ENTITY_SNAPSHOT_CACHE,
        memory_cache_size=config.ENTITY_SNAPSHOT_MEMORY_CACHE_SIZE,
        negative_ttl=config.ENTITY_SNAPSHOT_NEGATIVE_TTL_SECONDS,
        max_workers=config.SNAPSHOT_MAX_WORKERS,
        max_qps=config.SNAPSHOT_MAX_QPS,
        max_retries=config.SNAPSHOT_MAX_RETRIES,
    ):
        self.cache_db = Path(cache_db)
        self.enable_cache = enable_cache
        self.memory_cache_size = memory_cache_size
        self.negative_ttl = negative_ttl
        self.max_retries = max_retries
        self._memory_cache = OrderedDict()
        self._inflight = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._rate_limiter = RateLimiter(max_qps)
        self._snapshot_cache = SQLiteSnapshotCache(self.cache_db) if self.enable_cache else None
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "negative_hits": 0,
            "disk_hits": 0,
            "disk_writes": 0,
            "network_calls": 0,
            "network_errors": 0,
            "http_status_counts": {200: 0, 400: 0, 404: 0, 429: 0, "other": 0},
        }

    def _cache_key(self, qid, revision_id):
        return f"{qid}:{revision_id}"

    def _get_memory_cached(self, key):
        with self._lock:
            cached = self._memory_cache.get(key)
            if cached is None:
                return None
            self._memory_cache.move_to_end(key)
            self.stats["cache_hits"] += 1
            return cached

    def _store_memory(self, key, snapshot):
        if self.memory_cache_size <= 0:
            return
        with self._lock:
            self._memory_cache[key] = snapshot
            self._memory_cache.move_to_end(key)
            while len(self._memory_cache) > self.memory_cache_size:
                self._memory_cache.popitem(last=False)

    def _record_status(self, status_code):
        if status_code in self.stats["http_status_counts"]:
            self.stats["http_status_counts"][status_code] += 1
        else:
            self.stats["http_status_counts"]["other"] += 1

    def _encode_snapshot(self, snapshot):
        raw = json.dumps(snapshot, ensure_ascii=True).encode("utf-8")
        return zstd.ZstdCompressor().compress(raw)

    def _decode_snapshot(self, payload):
        raw = zstd.ZstdDecompressor().decompress(payload)
        return json.loads(raw.decode("utf-8"))

    def _negative_cache_valid(self, ts):
        if not ts or self.negative_ttl <= 0:
            return False
        cached_dt = parse_iso8601(ts)
        if not cached_dt:
            return False
        return datetime.now(timezone.utc) - cached_dt <= timedelta(seconds=self.negative_ttl)

    def _db_snapshot(self, key):
        if not self.enable_cache or not self._snapshot_cache:
            return None
        row = self._snapshot_cache.get(key)
        if not row:
            self.stats["cache_misses"] += 1
            return None
        status, payload, _content_type, ts = row
        if status == 200:
            if not payload:
                self._snapshot_cache.delete(key)
                self.stats["cache_misses"] += 1
                return None
            try:
                snapshot = self._decode_snapshot(payload)
            except Exception:
                self._snapshot_cache.delete(key)
                self.stats["cache_misses"] += 1
                return None
            self.stats["cache_hits"] += 1
            self.stats["disk_hits"] += 1
            return snapshot
        if self._negative_cache_valid(ts):
            self.stats["cache_hits"] += 1
            self.stats["negative_hits"] += 1
            return self._NEGATIVE_SENTINEL
        self._snapshot_cache.delete(key)
        self.stats["cache_misses"] += 1
        return None

    def _fetch_snapshot_network(self, qid, revision_id):
        endpoint = config.ENTITY_DATA_URL.format(qid=qid)
        params = {"revision": revision_id}
        for attempt in range(self.max_retries):
            self._rate_limiter.acquire()
            try:
                response = requests.get(endpoint, headers=config.HEADERS, params=params, timeout=config.API_TIMEOUT)
            except Exception:
                self.stats["network_errors"] += 1
                time.sleep(0.5 * (2**attempt))
                continue
            self.stats["network_calls"] += 1
            status = response.status_code
            content_type = response.headers.get("Content-Type")
            self._record_status(status)
            if status == 200:
                try:
                    return status, response.json(), content_type
                except Exception:
                    self.stats["network_errors"] += 1
                    return status, None, content_type
            if status in {429, 500, 502, 503, 504}:
                if attempt < self.max_retries - 1:
                    time.sleep(0.5 * (2**attempt))
                    continue
            self.stats["network_errors"] += 1
            return status, None, content_type
        self.stats["network_errors"] += 1
        return None, None, None

    def _parse_snapshot(self, data, qid):
        if not data or "entities" not in data:
            return None
        entity = data["entities"].get(qid)
        if not entity or "missing" in entity:
            return None
        return entity.get("claims", {})

    def _store_snapshot(self, key, status, snapshot, content_type):
        if not self.enable_cache or not self._snapshot_cache:
            return
        payload = self._encode_snapshot(snapshot) if status == 200 and snapshot is not None else None
        self._snapshot_cache.put(key, status, payload, content_type)
        self.stats["disk_writes"] += 1

    def _fetch_and_cache(self, qid, revision_id):
        key = self._cache_key(qid, revision_id)
        cached = self._get_memory_cached(key)
        if cached is not None:
            return cached
        snapshot = self._db_snapshot(key)
        if snapshot is not None or snapshot == self._NEGATIVE_SENTINEL:
            if snapshot is not None and snapshot is not self._NEGATIVE_SENTINEL:
                self._store_memory(key, snapshot)
            return None if snapshot == self._NEGATIVE_SENTINEL else snapshot

        status, data, content_type = self._fetch_snapshot_network(qid, revision_id)
        if status is None:
            return None
        if status == 200:
            snapshot = self._parse_snapshot(data, qid)
            if snapshot is None:
                self._store_snapshot(key, 404, None, content_type)
                return None
            self._store_memory(key, snapshot)
            self._store_snapshot(key, status, snapshot, content_type)
            return snapshot
        self._store_snapshot(key, status, None, content_type)
        return None

    def _clear_inflight(self, key):
        with self._lock:
            self._inflight.pop(key, None)

    def prefetch(self, qid, revision_ids, max_in_flight=None):
        if not revision_ids:
            return
        max_in_flight = max_in_flight if max_in_flight is not None else config.SNAPSHOT_PREFETCH
        for revision_id in revision_ids:
            key = self._cache_key(qid, revision_id)
            with self._lock:
                if key in self._inflight:
                    continue
                if len(self._inflight) >= max_in_flight:
                    break
            future = self._executor.submit(self._fetch_and_cache, qid, revision_id)
            with self._lock:
                if key in self._inflight:
                    continue
                self._inflight[key] = future
            future.add_done_callback(lambda _f, k=key: self._clear_inflight(k))

    def get_snapshot(self, qid, revision_id):
        key = self._cache_key(qid, revision_id)
        cached = self._get_memory_cached(key)
        if cached is not None:
            return cached
        snapshot = self._db_snapshot(key)
        if snapshot == self._NEGATIVE_SENTINEL:
            return None
        if snapshot is not None:
            self._store_memory(key, snapshot)
            return snapshot
        with self._lock:
            future = self._inflight.get(key)
        if future:
            return future.result()
        return self._fetch_and_cache(qid, revision_id)

    def inflight_size(self):
        with self._lock:
            return len(self._inflight)


REVISION_HISTORY_CACHE = RevisionHistoryCache() if config.ENABLE_HISTORY_CACHE else None
SNAPSHOT_FETCHER = SnapshotFetcher()


class LabelResolver:
    """Deterministic ID -> label resolution with SQLite caching."""

    def __init__(self, cache_path=config.LABEL_CACHE_DB, preferred_lang="en"):
        self.cache_path = Path(cache_path)
        self.preferred_lang = preferred_lang
        self.cache = SQLiteLabelCache(self.cache_path)
        self.failed_ids = set()
        self.stats = {
            "db_hits": 0,
            "db_misses": 0,
            "api_batches": 0,
            "api_ids": 0,
        }

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

        cached = self.cache.get_many(ordered_ids)
        self.stats["db_hits"] += len(cached)
        missing = [entity_id for entity_id in ordered_ids if entity_id not in cached]
        self.stats["db_misses"] += len(missing)
        if missing:
            fetched = self._fetch_and_cache(missing)
            cached.update(fetched)
        return {
            entity_id: {
                "label_en": cached.get(entity_id, (None, None))[0],
                "description_en": cached.get(entity_id, (None, None))[1],
            }
            for entity_id in ordered_ids
        }

    def lookup(self, entity_id):
        """Return cached resolution for a single id."""
        if not is_entity_or_property_id(entity_id):
            return self._null_entry()
        resolved = self.resolve([entity_id])
        return resolved.get(entity_id, self._null_entry())

    def _fetch_and_cache(self, ids):
        resolved_payload = {}
        for batch in chunked(ids, 50):
            params = {
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "props": "labels|descriptions",
            }
            self.stats["api_batches"] += 1
            self.stats["api_ids"] += len(batch)
            data = get_json(params)
            resolved_ids = set()
            updates = {}
            if data and "entities" in data:
                for entity_id, entity in data["entities"].items():
                    resolved_ids.add(entity_id)
                    if not entity or "missing" in entity:
                        updates[entity_id] = (None, None)
                        continue
                    updates[entity_id] = (
                        pick_label(entity, self.preferred_lang),
                        pick_description(entity, self.preferred_lang),
                    )
            unresolved = set(batch) - resolved_ids
            for missing_id in unresolved:
                if missing_id in self.failed_ids:
                    continue
                print(f"    [!] Warning: Unable to resolve {missing_id} via Wikidata API.")
                self.failed_ids.add(missing_id)
                updates[missing_id] = (None, None)
            if updates:
                self.cache.put_many(updates)
                resolved_payload.update(updates)
        return resolved_payload


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
    if config.ENABLE_HISTORY_CACHE and REVISION_HISTORY_CACHE:
        cached_revisions, cached_meta = REVISION_HISTORY_CACHE.get(qid, start_time, end_time)
        if cached_revisions is not None:
            return cached_revisions, cached_meta

    start_dt = parse_iso8601(start_time) if start_time else None
    end_dt = parse_iso8601(end_time) if end_time else None
    revisions = []
    carry_revision = None
    endpoint = config.REST_HISTORY_URL.format(qid=qid)
    next_endpoint = endpoint
    params = {"limit": 200}
    batches = 0
    truncated_by_window = False
    reached_page_limit = False
    api_calls = 0

    while next_endpoint and batches < config.MAX_HISTORY_PAGES:
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
            rev["_rev_dt"] = rev_dt
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
        if "_rev_dt" not in carry_revision:
            carry_revision["_rev_dt"] = parse_iso8601(carry_revision.get("timestamp"))
        revisions.append(carry_revision)

    def _revision_sort_key(rev):
        rev_dt = rev.get("_rev_dt")
        if isinstance(rev_dt, datetime):
            return rev_dt
        parsed = parse_iso8601(rev.get("timestamp"))
        if parsed is not None:
            return parsed
        return datetime.min.replace(tzinfo=timezone.utc)

    revisions.sort(key=_revision_sort_key)
    if next_endpoint and batches >= config.MAX_HISTORY_PAGES:
        reached_page_limit = True

    history_meta = {
        "qid": qid,
        "start_time": start_time,
        "end_time": end_time,
        "lookback_days": config.REVISION_LOOKBACK_DAYS,
        "max_history_pages": config.MAX_HISTORY_PAGES,
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
    history_meta["cache_hit"] = False

    if config.ENABLE_HISTORY_CACHE and REVISION_HISTORY_CACHE:
        REVISION_HISTORY_CACHE.store(qid, start_time, end_time, revisions, history_meta)

    return revisions, history_meta


def get_entity_snapshot(qid, revision_id):
    """Fetch and cache a full entity snapshot (claims dict) for a revision."""
    return SNAPSHOT_FETCHER.get_snapshot(qid, revision_id)


def get_claims_from_snapshot(snapshot, property_id):
    """Extract property claims from a snapshot claims dict."""
    if not snapshot:
        return []
    return snapshot.get(property_id, [])


def get_claims_for_revision(qid, property_id, revision_id):
    """Backward-compatible wrapper: fetch claims for a revision."""
    snapshot = get_entity_snapshot(qid, revision_id)
    return get_claims_from_snapshot(snapshot, property_id)


def extract_user(revision):
    """Normalize revision user data regardless of REST/Action response shape."""
    user = revision.get("user")
    if isinstance(user, dict):
        return user.get("name", "unknown")
    return user or "unknown"
