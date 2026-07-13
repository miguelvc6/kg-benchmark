import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import fetcher
from lib import config


class FetcherPathTests(unittest.TestCase):
    def test_explicit_runtime_paths_are_isolated(self) -> None:
        original = (config.DATA_DIR, config.CACHE_DIR, config.LATEST_DUMP_PATH)
        with tempfile.TemporaryDirectory() as temporary, patch("lib.caching.SnapshotFetcher", return_value=Mock()), patch(
            "lib.caching.RevisionHistoryCache", return_value=Mock()
        ):
            try:
                root = Path(temporary) / "snapshot"
                cache = Path(temporary) / "cache"
                dump = Path(temporary) / "dump.json.bz2"
                resolved = fetcher.configure_runtime_paths(data_dir=root, cache_dir=cache, dump_path=dump)
                self.assertEqual(resolved, {"data_dir": root, "cache_dir": cache, "dump_path": dump})
                self.assertEqual(fetcher.WIKIDATA_REPAIRS, root / "02_wikidata_repairs.json")
                self.assertEqual(fetcher.REPAIR_CANDIDATES_FILE, root / "01_repair_candidates.json")
                self.assertEqual(config.ENTITY_SNAPSHOT_DB, cache / "entity_snapshots.sqlite")
                self.assertEqual(fetcher.LATEST_DUMP_PATH, dump)
            finally:
                fetcher.configure_runtime_paths(data_dir=original[0], cache_dir=original[1], dump_path=original[2])


if __name__ == "__main__":
    unittest.main()
