import re
from datetime import UTC, datetime
from pathlib import Path

# HTTP identity and base endpoints
HEADERS = {"User-Agent": "WikidataRepairEval/1.0 (PhD Research; mailto:miguel.vazquez@wu.ac.at)"}
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
REPORT_HISTORY_DEPTH = 20  # Revision pairs scanned per report page

# Snapshot cache and throttling controls
ENABLE_ENTITY_SNAPSHOT_CACHE = True
ENTITY_SNAPSHOT_CACHE_DIR = Path("data/cache/entity_snapshots")
ENTITY_SNAPSHOT_NEGATIVE_TTL_SECONDS = 300
ENTITY_SNAPSHOT_MEMORY_CACHE_SIZE = 512
SNAPSHOT_MAX_WORKERS = 32
SNAPSHOT_MAX_QPS = 10
SNAPSHOT_PREFETCH = 6
SNAPSHOT_MAX_RETRIES = 4

# History cache sizing
ENABLE_HISTORY_CACHE = True
HISTORY_CACHE_MAX_ENTRIES = 20000
HISTORY_CACHE_MAX_SEGMENTS_PER_QID = 4

# Report parsing and ID validation patterns
QID_PATTERN = re.compile(r"\[\[(Q\d+)\]\]")
REPORT_VIOLATION_QID_PATTERN = re.compile(r"Q\|?(\d+)")
QID_EXACT_PATTERN = re.compile(r"^Q\d+$")
PID_EXACT_PATTERN = re.compile(r"^P\d+$")
INVALID_REPORT_SECTIONS = {
    "Types statistics",
}

# Input/output locations and core data artifacts
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LATEST_DUMP_PATH = Path("data/latest-all.json.gz")  # 2026 dump location
WORLD_STATE_FILE = Path("data/03_world_state.json")  # Output for built contexts
LABEL_CACHE_FILE = CACHE_DIR / "id_labels_en.json"
LABEL_CACHE_DB = CACHE_DIR / "labels_en.sqlite"
ENTITY_SNAPSHOT_DB = CACHE_DIR / "entity_snapshots.sqlite"
REPAIR_CANDIDATES_FILE = DATA_DIR / "01_repair_candidates.json"
WIKIDATA_REPAIRS = DATA_DIR / "02_wikidata_repairs.json"
WIKIDATA_REPAIRS_JSONL = DATA_DIR / "02_wikidata_repairs.jsonl"

# Popularity and pageview enrichment configuration
POPULARITY_FILE = DATA_DIR / "00_entity_popularity.json"
PAGEVIEWS_CACHE_FILE = CACHE_DIR / "pageviews_enwiki_365d.json"
POPULARITY_WINDOW_DAYS = 365
POPULARITY_WIKI = "enwiki"
PAGEVIEWS_PROJECT = "en.wikipedia.org"
PAGEVIEWS_ACCESS = "all-access"
PAGEVIEWS_AGENT = "user"
PAGEVIEWS_GRANULARITY = "daily"
PAGEVIEWS_ENDPOINT = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "{project}/{access}/{agent}/{article}/{granularity}/{start}/{end}"
)
POPULARITY_WEIGHTS = {
    "pageviews_norm": 0.5,
    "degree_norm": 0.3,
    "sitelinks_norm": 0.2,
}

# Optional debug target list; leave empty to process all properties
TARGET_PROPERTIES = [
    # "P569",  # Date of Birth
    # "P570",  # Date of Death
    # "P21",  # Sex or Gender
]

# Run logging and telemetry
RUN_ID = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
STATS_FILE = LOG_DIR / f"fetcher_stats_{RUN_ID}.jsonl"
SUMMARY_FILE = LOG_DIR / f"run_summary_{RUN_ID}.json"
STATS_FLUSH_EVERY = 10000
STAGE2_LOG_EVERY = 1000
RESUME_CHECKPOINT_EVERY = 5000
RESUME_DEFAULT_CHECKPOINT = LOG_DIR / f"resume_checkpoint_{RUN_ID}.json"

# Labels and world-state validation constants
MISSING_LABEL_PLACEHOLDER = "Label unavailable"
REQUIRED_WORLD_STATE_KEYS = ("L1_ego_node", "L2_labels", "L3_neighborhood", "L4_constraints")
