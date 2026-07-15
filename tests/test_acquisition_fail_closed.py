import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import requests

import fetcher
from lib import config
from lib.caching import SnapshotFetcher, SnapshotFetchError
from lib.mining import ensure_repair_candidates_file, mine_repairs
from lib.popularity import PageviewClient
from lib.utils import TransientAPIError, get_json
from lib.world_state import WorldStateBuilder


class AcquisitionFailClosedTests(unittest.TestCase):
    def test_partial_stage2_jsonl_requires_explicit_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            partial = root / "02_wikidata_repairs.jsonl"
            partial.write_text(json.dumps({"id": "partial"}) + "\n", encoding="utf-8")
            with (
                patch.object(fetcher, "REPAIR_CANDIDATES_FILE", root / "01_repair_candidates.json"),
                patch.object(fetcher, "WIKIDATA_REPAIRS", root / "02_wikidata_repairs.json"),
                patch.object(fetcher, "WIKIDATA_REPAIRS_JSONL", partial),
                patch.object(fetcher, "ensure_repair_candidates_file", return_value=[{"qid": "Q1"}]),
                patch.object(fetcher, "deduplicate_candidates", return_value=([{"qid": "Q1"}], {})),
                patch.object(fetcher, "LabelResolver"),
                patch.object(fetcher, "load_cached_repairs", return_value=None),
                self.assertRaisesRegex(RuntimeError, "may be partial output"),
            ):
                fetcher.process_pipeline()

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
            patch("lib.mining._wait_for_report_request_slot"),
            patch("lib.mining.random.uniform", return_value=0.0),
            patch("lib.mining.time.sleep") as sleep,
            self.assertRaisesRegex(RuntimeError, "Failed to fetch report page"),
        ):
            mine_repairs("P1", max_items=2)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5.0, 10.0, 20.0, 40.0, 80.0])

    def test_report_page_retry_honors_retry_after_header(self) -> None:
        response = requests.Response()
        response.status_code = 429
        response.headers["Retry-After"] = "17"
        error = requests.HTTPError("rate limited", response=response)
        page = Mock()
        page.revisions.side_effect = [error, []]
        site = MagicMock()
        site.pages.__getitem__.return_value = page
        with (
            patch("lib.mining.get_wikidata_site", return_value=site),
            patch("lib.mining._wait_for_report_request_slot"),
            patch("lib.mining.random.uniform", return_value=0.25),
            patch("lib.mining.time.sleep") as sleep,
        ):
            self.assertEqual(mine_repairs("P1", max_items=2), [])
        sleep.assert_called_once_with(17.25)

    def test_report_page_lookup_is_throttled_and_retried(self) -> None:
        response = requests.Response()
        response.status_code = 429
        response.headers["Retry-After"] = "9"
        error = requests.HTTPError("rate limited", response=response)
        page = Mock()
        page.revisions.return_value = []
        site = MagicMock()
        site.pages.__getitem__.side_effect = [error, page]
        with (
            patch("lib.mining.get_wikidata_site", return_value=site),
            patch("lib.mining._wait_for_report_request_slot") as throttle,
            patch("lib.mining.random.uniform", return_value=0.0),
            patch("lib.mining.time.sleep") as sleep,
        ):
            self.assertEqual(mine_repairs("P1", max_items=2), [])
        sleep.assert_called_once_with(9.0)
        self.assertEqual(throttle.call_count, 3)

    def test_stage1_checkpoint_resumes_after_last_completed_property(self) -> None:
        first_candidate = {
            "qid": "Q1",
            "property_id": "P1",
            "violation_type": "Single value",
            "fix_date": "2026-01-01T00:00:00",
            "report_revision_old": 1,
            "report_revision_new": 2,
        }
        second_candidate = {
            "qid": "Q2",
            "property_id": "P2",
            "violation_type": "Format",
            "fix_date": "2026-01-02T00:00:00",
            "report_revision_old": 3,
            "report_revision_new": 4,
        }
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "01_repair_candidates.json"
            with (
                patch.object(config, "TARGET_PROPERTIES", ["P1", "P2"]),
                patch("lib.mining.mine_repairs", side_effect=[[first_candidate], OSError("interrupted")]),
                self.assertRaisesRegex(OSError, "interrupted"),
            ):
                ensure_repair_candidates_file(target, history_limit=2, force_refresh=True)

            checkpoint = target.with_name("01_repair_candidates.checkpoint.json")
            partial = target.with_name("01_repair_candidates.partial.jsonl")
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(partial.is_file())
            self.assertFalse(target.exists())
            with partial.open("ab") as fh:
                fh.write(b'{"uncheckpointed":')

            with (
                patch.object(config, "TARGET_PROPERTIES", []),
                patch("lib.mining.mine_repairs", return_value=[second_candidate]) as mine,
            ):
                candidates = ensure_repair_candidates_file(target, history_limit=2, force_refresh=True)

            mine.assert_called_once_with("P2", max_items=2)
            self.assertEqual(candidates, [first_candidate, second_candidate])
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), candidates)
            self.assertFalse(checkpoint.exists())
            self.assertFalse(partial.exists())


if __name__ == "__main__":
    unittest.main()
