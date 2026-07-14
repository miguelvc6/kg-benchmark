import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from lib.caching import SnapshotFetcher, SnapshotFetchError
from lib.mining import mine_repairs
from lib.popularity import PageviewClient
from lib.utils import TransientAPIError, get_json
from lib.world_state import WorldStateBuilder


class AcquisitionFailClosedTests(unittest.TestCase):
    def test_json_api_exhaustion_raises_when_required(self) -> None:
        with (
            patch("lib.utils.requests.get", side_effect=OSError("offline")),
            patch("lib.utils.time.sleep"),
            self.assertRaises(TransientAPIError),
        ):
            get_json(endpoint="https://example.invalid/api", raise_on_failure=True)

    def test_snapshot_transient_exhaustion_is_not_missing_data(self) -> None:
        fetcher = SnapshotFetcher(enable_cache=False, max_retries=2, max_qps=0)
        with (
            patch("lib.caching.requests.get", side_effect=OSError("offline")),
            patch("lib.caching.time.sleep"),
            self.assertRaises(SnapshotFetchError),
        ):
            fetcher.get_snapshot("Q1", 123)

    def test_snapshot_terminal_404_remains_a_negative_result(self) -> None:
        response = Mock(status_code=404, headers={})
        fetcher = SnapshotFetcher(enable_cache=False, max_retries=2, max_qps=0)
        with patch("lib.caching.requests.get", return_value=response):
            self.assertIsNone(fetcher.get_snapshot("Q1", 123))

    def test_corrupt_dump_cannot_produce_partial_world_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dump = Path(temporary) / "broken.json.gz"
            dump.write_bytes(b"not-a-gzip-stream")
            builder = WorldStateBuilder(dump)
            with self.assertRaisesRegex(RuntimeError, "partial context is invalid"):
                builder._load_entities_from_dump({"Q1"})

    def test_pageview_transient_failure_is_not_zero_popularity(self) -> None:
        response = Mock(status_code=503)
        with tempfile.TemporaryDirectory() as temporary:
            client = PageviewClient(cache_path=Path(temporary) / "pageviews.json")
            with (
                patch("lib.popularity.requests.get", return_value=response),
                patch("lib.popularity.time.sleep"),
                self.assertRaisesRegex(RuntimeError, "exhausted retries"),
            ):
                client._fetch_article_pageviews("Example", "2025010100", "2026010100")

    def test_report_page_exhaustion_aborts_candidate_refresh(self) -> None:
        page = Mock()
        page.revisions.side_effect = OSError("offline")
        site = MagicMock()
        site.pages.__getitem__.return_value = page
        with (
            patch("lib.mining.get_wikidata_site", return_value=site),
            patch("lib.mining.time.sleep"),
            self.assertRaisesRegex(RuntimeError, "Failed to fetch report page"),
        ):
            mine_repairs("P1", max_items=2)


if __name__ == "__main__":
    unittest.main()
