import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import requests
from jsonschema import Draft202012Validator

import fetcher
from kg_benchmark import cli as benchmark_cli
from lib import config
from lib.caching import SnapshotFetcher, SnapshotFetchError, fetch_revision_history
from lib.mining import (
    build_report_provenance,
    ensure_repair_candidates_file,
    invalid_report_transition_reason,
    mine_repairs,
    sample_candidates_by_report_event,
)
from lib.popularity import PageviewClient
from lib.utils import TerminalAPIError, TransientAPIError, get_json
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

    def test_json_api_terminal_404_is_not_retried(self) -> None:
        response = Mock(status_code=404)
        with (
            patch("lib.utils.requests.get", return_value=response) as request,
            patch("lib.utils.time.sleep") as sleep,
            self.assertRaisesRegex(TerminalAPIError, "terminal HTTP 404"),
        ):
            get_json(endpoint="https://example.invalid/missing", raise_on_failure=True)
        request.assert_called_once()
        sleep.assert_not_called()

    def test_missing_entity_history_returns_terminal_metadata(self) -> None:
        error = TerminalAPIError("https://example.invalid/Q1/history", 404)
        with (
            patch("lib.caching.get_json", side_effect=error),
            patch.object(config, "ENABLE_HISTORY_CACHE", False),
        ):
            revisions, metadata = fetch_revision_history(
                "Q1", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"
            )
        self.assertEqual(revisions, [])
        self.assertTrue(metadata["terminal_missing"])
        self.assertEqual(metadata["api_calls"], 0)

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

    def test_stage2_transient_exhaustion_becomes_explicit_exclusion(self) -> None:
        def fail_snapshot():
            raise SnapshotFetchError("Historical snapshot fetch exhausted retries for P1@123")

        result, failure = fetcher._run_stage2_api_phase(
            "synthetic Stage-2 phase",
            "t_box_history",
            fail_snapshot,
        )
        self.assertIsNone(result)
        self.assertEqual(failure["phase"], "t_box_history")
        self.assertEqual(failure["error_type"], "SnapshotFetchError")

        record = fetcher._build_stage2_exclusion_record(
            {
                "qid": "Q1",
                "property": "P1",
                "violation_type": "Format",
                "candidate_key": "Q1|P1|2026-01-01T00:00:00|1|2",
                "candidate_index": 7,
                "fix_date": "2026-01-01T00:00:00",
                "report_revision_old": 1,
                "report_revision_new": 2,
                "report_event_sampling": None,
            },
            failure,
            run_id="test-run",
        )
        schema_path = Path(__file__).resolve().parents[1] / "schemas" / "stage2-candidate-exclusion.schema.json"
        Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(record)
        self.assertEqual(record["result"], "upstream_unavailable")
        self.assertEqual(record["exclusion_reason"], "transient_api_retry_exhausted")

    def test_cross_freeze_resume_binds_completed_prefix_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "acquisition"
            data_dir.mkdir()
            (data_dir / "01_repair_candidates.json").write_text("[]\n", encoding="utf-8")
            (data_dir / "02_wikidata_repairs.jsonl").write_text(
                json.dumps({"id": "repair_Q1_2"}) + "\n",
                encoding="utf-8",
            )
            stats = data_dir / "fetcher_stats.jsonl"
            stats.write_text(
                json.dumps(
                    {
                        "candidate_key": "Q1|P1|2026-01-01T00:00:00|1|2",
                        "candidate_index": 0,
                        "result": "no_diff",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = root / "acquisition-config.json"
            prior_methodology = {
                "freeze_scope_sha256": "a" * 64,
                "source_git_revision": "b" * 40,
                "methodology_lock_sha256": "c" * 64,
            }
            config_path.write_text(
                json.dumps(
                    {
                        "manifest_type": "dataset_acquisition",
                        "manifest_version": 1,
                        "status": "failed",
                        "methodology": prior_methodology,
                    }
                ),
                encoding="utf-8",
            )

            provenance = benchmark_cli._build_resume_provenance(
                [
                    "--data-dir",
                    str(data_dir),
                    "--resume-stats",
                    str(stats),
                ],
                config_path=config_path,
                current_methodology={"lock": {"source_git_revision": "d" * 40}},
            )

        self.assertTrue(provenance["methodology_changed"])
        self.assertEqual(provenance["prior_methodology"], prior_methodology)
        self.assertEqual(provenance["artifacts"]["resume_stats"]["records"], 1)
        self.assertTrue(
            provenance["artifacts"]["resume_stats"]["candidate_indices_contiguous_segment"]
        )

    def test_process_pipeline_skips_transient_candidate_and_checkpoints_it(self) -> None:
        candidate = {
            "qid": "Q1",
            "property_id": "P1",
            "violation_type": "Format",
            "fix_date": "2026-01-01T00:00:00",
            "report_revision_old": 1,
            "report_revision_new": 2,
            "report_event_sampling": None,
        }
        event_stats = {
            "method": "sha256_qid_rank_v1",
            "seed": 13,
            "cap": 100,
            "events": 1,
            "capped_events": 0,
            "candidates_removed": 0,
            "post_cap_candidates": 1,
            "pre_cap_candidates": 1,
            "reused_sampled_artifact": True,
        }
        dedup_stats = {
            "duplicates_skipped": 0,
            "violation_type_merges": 0,
            "exact_duplicates": 0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repairs_json = root / "02_wikidata_repairs.json"
            repairs_jsonl = root / "02_wikidata_repairs.jsonl"
            exclusions_json = root / "02_stage2_exclusions.json"
            exclusions_jsonl = root / "02_stage2_exclusions.jsonl"
            stats_path = root / "logs" / "stats.jsonl"
            summary_path = root / "logs" / "summary.json"
            checkpoint_path = root / "logs" / "checkpoint.json"
            label_resolver = Mock(stats={"db_hits": 0, "db_misses": 0})
            with (
                patch.object(fetcher, "REPAIR_CANDIDATES_FILE", root / "01_repair_candidates.json"),
                patch.object(fetcher, "WIKIDATA_REPAIRS", repairs_json),
                patch.object(fetcher, "WIKIDATA_REPAIRS_JSONL", repairs_jsonl),
                patch.object(fetcher, "STAGE2_EXCLUSIONS", exclusions_json),
                patch.object(fetcher, "STAGE2_EXCLUSIONS_JSONL", exclusions_jsonl),
                patch.object(fetcher, "STATS_FILE", stats_path),
                patch.object(fetcher, "SUMMARY_FILE", summary_path),
                patch.object(fetcher, "RESUME_DEFAULT_CHECKPOINT", checkpoint_path),
                patch.object(fetcher, "ensure_repair_candidates_file", return_value=[candidate]),
                patch.object(
                    fetcher,
                    "deduplicate_candidates",
                    return_value=([candidate], dedup_stats),
                ),
                patch.object(
                    fetcher,
                    "sample_candidates_by_report_event",
                    return_value=([candidate], event_stats),
                ),
                patch.object(fetcher, "LabelResolver", return_value=label_resolver),
                patch.object(fetcher, "load_cached_repairs", side_effect=[None, []]),
                patch.object(
                    fetcher,
                    "find_repair_revision",
                    side_effect=SnapshotFetchError("synthetic timeout"),
                ),
            ):
                fetcher.process_pipeline()

            exclusions = json.loads(exclusions_json.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            stats_rows = [
                json.loads(line)
                for line in stats_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(len(exclusions), 1)
        self.assertEqual(exclusions[0]["result"], "upstream_unavailable")
        self.assertEqual(exclusions[0]["phase"], "entity_history")
        self.assertEqual(summary["upstream_unavailable_total"], 1)
        self.assertEqual(summary["no_diff"], 0)
        self.assertEqual(summary["no_history"], 0)
        self.assertEqual(stats_rows[0]["candidate_key"], exclusions[0]["candidate_key"])
        self.assertTrue(checkpoint["completed"])
        self.assertEqual(checkpoint["last_index"], 0)
        self.assertEqual(checkpoint["upstream_unavailable_total"], 1)

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

    def test_report_error_revision_does_not_generate_candidates(self) -> None:
        older = {
            "*": "== Allowed qualifiers ==\n[[Q110404588]]",
            "revid": 2461550294,
            "timestamp": (2026, 2, 7, 16, 17, 23),
            "size": 521775,
            "comment": "report update for [[Property:P166]]",
        }
        newer = {
            "*": "Report update failed",
            "revid": 2462103772,
            "timestamp": (2026, 2, 9, 20, 57, 2),
            "size": 108,
            "comment": "error while update for [[Property:P166]]",
        }
        page = Mock()
        page.revisions.return_value = [newer, older]
        site = MagicMock()
        site.pages.__getitem__.return_value = page
        with (
            patch("lib.mining.get_wikidata_site", return_value=site),
            patch("lib.mining._wait_for_report_request_slot"),
        ):
            self.assertEqual(mine_repairs("P166", max_items=2), [])
        self.assertIn("comment", page.revisions.call_args.kwargs["prop"])
        self.assertIn("size", page.revisions.call_args.kwargs["prop"])

    def test_unmarked_tiny_report_collapse_is_rejected(self) -> None:
        old_map = {f"Q{qid}": {"Format"} for qid in range(1, 101)}
        older = {"*": "full report", "size": 100000, "comment": "report update"}
        newer = {"*": "small error page", "size": 500, "comment": "report update"}
        self.assertEqual(
            invalid_report_transition_reason(older, newer, old_map, {}),
            "suspicious_tiny_report_collapse",
        )

    def test_normal_report_disappearance_remains_a_candidate(self) -> None:
        older = {
            "*": "== Format ==\n[[Q1]]",
            "revid": 10,
            "timestamp": (2026, 1, 1, 0, 0, 0),
            "size": 5000,
            "comment": "report update for [[Property:P1]]",
        }
        newer = {
            "*": "== Format ==\n",
            "revid": 11,
            "timestamp": (2026, 1, 2, 0, 0, 0),
            "size": 4900,
            "comment": "report update for [[Property:P1]]",
        }
        page = Mock()
        page.revisions.return_value = [newer, older]
        site = MagicMock()
        site.pages.__getitem__.return_value = page
        with (
            patch("lib.mining.get_wikidata_site", return_value=site),
            patch("lib.mining._wait_for_report_request_slot"),
        ):
            candidates = mine_repairs("P1", max_items=2)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["qid"], "Q1")

    def test_report_event_sampling_is_deterministic_and_idempotent(self) -> None:
        large_event = [
            {
                "qid": f"Q{qid}",
                "property_id": "P1",
                "violation_type": "Format",
                "fix_date": "2026-01-02T00:00:00",
                "report_revision_old": 10,
                "report_revision_new": 11,
            }
            for qid in range(1, 151)
        ]
        small_event = [
            {
                "qid": f"Q{qid}",
                "property_id": "P2",
                "violation_type": "Single value",
                "fix_date": "2026-01-03T00:00:00",
                "report_revision_old": 20,
                "report_revision_new": 21,
            }
            for qid in range(151, 154)
        ]
        sampled, stats = sample_candidates_by_report_event(large_event + small_event, cap=100, seed=13)
        reversed_sampled, _ = sample_candidates_by_report_event(
            list(reversed(large_event + small_event)), cap=100, seed=13
        )

        self.assertEqual([row["qid"] for row in sampled], [row["qid"] for row in reversed_sampled])
        self.assertEqual(len(sampled), 103)
        self.assertEqual(stats["events"], 2)
        self.assertEqual(stats["capped_events"], 1)
        self.assertEqual(stats["candidates_removed"], 50)

        large_rows = [row for row in sampled if row["property_id"] == "P1"]
        self.assertEqual(len(large_rows), 100)
        self.assertEqual(
            [row["report_event_sampling"]["rank"] for row in large_rows],
            list(range(1, 101)),
        )
        self.assertTrue(all(row["report_event_sampling"]["event_candidate_count"] == 150 for row in large_rows))
        self.assertTrue(all(row["report_event_sampling"]["capped"] for row in large_rows))

        reused, reused_stats = sample_candidates_by_report_event(sampled, cap=100, seed=13)
        self.assertEqual(reused, sampled)
        self.assertTrue(reused_stats["reused_sampled_artifact"])
        self.assertEqual(reused_stats["pre_cap_candidates"], 153)

    def test_report_event_sampling_propagates_to_report_provenance(self) -> None:
        candidate = {
            "qid": "Q1",
            "property_id": "P1",
            "violation_type": "Format",
            "fix_date": "2026-01-02T00:00:00",
            "report_revision_old": 10,
            "report_revision_new": 11,
        }
        sampled, _ = sample_candidates_by_report_event([candidate], cap=100, seed=13)
        provenance = build_report_provenance(sampled[0], "P1")
        self.assertEqual(provenance["report_event_sampling"], sampled[0]["report_event_sampling"])

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
