import copy
import json
import math
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import requests

from . import config
from .utils import is_qid
from .world_state import WorldStateBuilder


class PageviewClient:
    """Local cache around the Wikimedia pageviews API."""

    def __init__(
        self,
        cache_path=config.PAGEVIEWS_CACHE_FILE,
        project=config.PAGEVIEWS_PROJECT,
        access=config.PAGEVIEWS_ACCESS,
        agent=config.PAGEVIEWS_AGENT,
        granularity=config.PAGEVIEWS_GRANULARITY,
    ):
        self.cache_path = Path(cache_path)
        self.project = project
        self.access = access
        self.agent = agent
        self.granularity = granularity
        self.cache = {}
        self._dirty = False
        self._load_cache()

    def _load_cache(self):
        if not self.cache_path.exists():
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            print(f"[!] Warning: Could not load pageview cache {self.cache_path}: {exc}")
            return
        if isinstance(data, dict):
            self.cache.update(data)

    def persist(self):
        if not self._dirty:
            return
        ordered = {key: self.cache[key] for key in sorted(self.cache.keys())}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as fh:
            json.dump(ordered, fh, ensure_ascii=True, indent=2)
        self._dirty = False

    def _signature(self, article_title, start, end):
        return {
            "article": article_title,
            "start": start,
            "end": end,
            "project": self.project,
            "access": self.access,
            "agent": self.agent,
            "granularity": self.granularity,
        }

    @staticmethod
    def _signatures_match(entry, signature):
        for key, value in signature.items():
            if entry.get(key) != value:
                return False
        return True

    def get_total_pageviews(self, qid, article_title, start, end):
        """Return cached or freshly fetched 365d totals for the article."""
        signature = self._signature(article_title, start, end)
        cached = self.cache.get(qid)
        if cached and self._signatures_match(cached, signature):
            return cached.get("views", 0)
        total = self._fetch_article_pageviews(article_title, start, end)
        payload = dict(signature)
        payload["views"] = total
        self.cache[qid] = payload
        self._dirty = True
        return total

    def _fetch_article_pageviews(self, article_title, start, end):
        if not article_title:
            return 0
        safe_article = quote(article_title.replace(" ", "_"), safe="")
        url = config.PAGEVIEWS_ENDPOINT.format(
            project=self.project,
            access=self.access,
            agent=self.agent,
            article=safe_article,
            granularity=self.granularity,
            start=start,
            end=end,
        )
        for attempt in range(4):
            try:
                response = requests.get(url, headers=config.HEADERS, timeout=config.API_TIMEOUT)
            except Exception as exc:
                print(f"    [!] Pageviews request failed for {article_title}: {exc}")
                time.sleep(0.5)
                continue
            status = response.status_code
            if status == 200:
                data = response.json()
                items = data.get("items", [])
                return sum(item.get("views", 0) or 0 for item in items)
            if status in {204, 404}:
                return 0
            sleep_for = 2**attempt
            print(f"    [!] Pageviews HTTP {status} for {article_title}. Sleeping {sleep_for}s...")
            time.sleep(sleep_for)
        print(f"    [!] Pageviews permanently failed for {article_title}. Defaulting to 0.")
        return 0


def _percentile_map(value_pairs):
    """Return percentile ranks for (value, qid) tuples."""
    if not value_pairs:
        return {}
    sorted_pairs = sorted(value_pairs, key=lambda pair: pair[0])
    n = len(sorted_pairs)
    if n == 1:
        return {sorted_pairs[0][1]: 1.0}
    ranks = {}
    idx = 0
    while idx < n:
        value = sorted_pairs[idx][0]
        start_idx = idx
        while idx < n and sorted_pairs[idx][0] == value:
            idx += 1
        end_idx = idx - 1
        percentile = (start_idx + end_idx) / 2 / (n - 1)
        for cursor in range(start_idx, end_idx + 1):
            ranks[sorted_pairs[cursor][1]] = percentile
    return ranks


class PopularityCalculator:
    """Compute composite popularity scores for a list of QIDs."""

    def __init__(
        self,
        dump_path=config.LATEST_DUMP_PATH,
        window_days=config.POPULARITY_WINDOW_DAYS,
        wiki=config.POPULARITY_WIKI,
        pageview_client=None,
    ):
        self.dump_path = Path(dump_path)
        self.window_days = window_days
        self.wiki = wiki
        self.pageviews = pageview_client or PageviewClient()
        self.build_datetime = datetime.now(UTC)

    def build(self, qids):
        focus_ids = [qid for qid in sorted(set(qids)) if is_qid(qid)]
        if not focus_ids:
            return {}
        entity_loader = WorldStateBuilder(self.dump_path)
        if not entity_loader.has_dump:
            raise RuntimeError(f"Wikidata dump not found at {self.dump_path}. It is required for popularity scoring.")
        focus_entities = entity_loader._load_entities_from_dump(set(focus_ids))
        missing = set(focus_ids) - set(focus_entities.keys())
        if missing:
            print(f"    [!] Warning: Popularity calculation missing {len(missing)} focus entities in dump.")
        start_str, end_str = self._window_bounds()
        raw_components = {}
        log_pairs = {"pageviews": [], "degree": [], "sitelinks": []}
        for qid in focus_ids:
            entity = focus_entities.get(qid)
            components = self._compute_components(qid, entity, start_str, end_str)
            raw_components[qid] = components
            log_pairs["pageviews"].append((math.log1p(components["pageviews_365d"]), qid))
            log_pairs["degree"].append((math.log1p(components["out_degree"]), qid))
            log_pairs["sitelinks"].append((math.log1p(components["sitelinks_count"]), qid))

        pageviews_norm = _percentile_map(log_pairs["pageviews"])
        degree_norm = _percentile_map(log_pairs["degree"])
        sitelinks_norm = _percentile_map(log_pairs["sitelinks"])

        popularity_map = {}
        for qid, components in raw_components.items():
            normalized = {
                "pageviews_norm": pageviews_norm.get(qid, 0.0),
                "degree_norm": degree_norm.get(qid, 0.0),
                "sitelinks_norm": sitelinks_norm.get(qid, 0.0),
            }
            score = (
                config.POPULARITY_WEIGHTS["pageviews_norm"] * normalized["pageviews_norm"]
                + config.POPULARITY_WEIGHTS["degree_norm"] * normalized["degree_norm"]
                + config.POPULARITY_WEIGHTS["sitelinks_norm"] * normalized["sitelinks_norm"]
            )
            popularity_map[qid] = {
                "score": score,
                "components": {
                    **components,
                    **normalized,
                },
            }
        self.pageviews.persist()
        return popularity_map

    def _window_bounds(self):
        end_dt = self.build_datetime
        start_dt = end_dt - timedelta(days=self.window_days)
        start_str = start_dt.strftime("%Y%m%d00")
        end_str = end_dt.strftime("%Y%m%d00")
        return start_str, end_str

    def _compute_components(self, qid, entity, start, end):
        pageviews_total = self.pageviews.get_total_pageviews(qid, self._enwiki_title(entity), start, end)
        return {
            "pageviews_365d": int(pageviews_total),
            "out_degree": self._out_degree(entity),
            "sitelinks_count": len(entity.get("sitelinks", {})) if entity else 0,
            "pageviews_project": self.pageviews.project,
            "pageviews_access": self.pageviews.access,
            "pageviews_agent": self.pageviews.agent,
            "pageviews_granularity": self.pageviews.granularity,
            "pageviews_start": start,
            "pageviews_end": end,
            "wiki": self.wiki,
        }

    @staticmethod
    def _enwiki_title(entity):
        if not entity:
            return None
        sitelinks = entity.get("sitelinks", {})
        if not sitelinks:
            return None
        enwiki = sitelinks.get("enwiki")
        if not enwiki:
            return None
        return enwiki.get("title")

    @staticmethod
    def _out_degree(entity):
        if not entity:
            return 0
        claims = entity.get("claims", {})
        total = 0
        for statements in claims.values():
            total += len(statements)
        return total


def load_popularity_artifact(path=config.POPULARITY_FILE):
    """Return on-disk popularity map if available."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        return None
    try:
        with open(artifact_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        print(f"[!] Warning: Could not read popularity artifact {artifact_path}: {exc}")
        return None
    if isinstance(payload, dict) and payload:
        return payload
    print(f"[!] Warning: Popularity artifact {artifact_path} malformed. Recomputing.")
    return None


def persist_popularity_artifact(popularity_map, path=config.POPULARITY_FILE):
    """Persist QID -> popularity dictionary using deterministic ordering."""
    artifact_path = Path(path)
    ordered = {qid: popularity_map[qid] for qid in sorted(popularity_map.keys())}
    with open(artifact_path, "w", encoding="utf-8") as fh:
        json.dump(ordered, fh, ensure_ascii=True, indent=2)


def ensure_entity_popularity(entries):
    """Return popularity map aligned with the provided Stage-2 entries."""
    qids = [entry.get("qid") for entry in entries if isinstance(entry, dict) and entry.get("qid")]
    cached = load_popularity_artifact()
    if cached:
        cached_ids = set(cached.keys())
        if cached_ids == set(qids):
            print(f"[+] Using cached popularity artifact at {config.POPULARITY_FILE}.")
            return cached
        print("[*] Popularity artifact out of date. Recomputing to maintain deterministic percentiles.")
    calculator = PopularityCalculator()
    popularity_map = calculator.build(qids)
    persist_popularity_artifact(popularity_map)
    return popularity_map


def attach_entity_popularity(entries, popularity_lookup):
    """Attach per-qid popularity blocks to Stage-2 entries."""
    if not entries or not popularity_lookup:
        return entries
    for entry in entries:
        qid = entry.get("qid")
        if not qid:
            continue
        block = popularity_lookup.get(qid)
        if block:
            entry["popularity"] = copy.deepcopy(block)
    return entries
