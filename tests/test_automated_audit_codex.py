import hashlib
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from automated_audit_codex import (
    DISAGREEMENTS_FILENAME,
    DISPOSITIONS_FILENAME,
    FINAL_JSON_FILENAME,
    FINAL_MD_FILENAME,
    REVIEWS_FILENAME,
    RUN_FILENAME,
    AutomatedAuditError,
    finalize_audit,
    run_codex_reviews,
)


class FakeCodex:
    def __init__(self, *, invalid_first: bool = False, omit_last: bool = False) -> None:
        self.calls: list[dict] = []
        self.invalid_first = invalid_first
        self.omit_last = omit_last
        self._exec_count = 0
        self._lock = threading.Lock()

    def __call__(self, command, **kwargs):
        if command == ["codex", "--version"]:
            return subprocess.CompletedProcess(command, 0, "codex-cli test-version\n", "")
        with self._lock:
            self._exec_count += 1
            exec_count = self._exec_count
            self.calls.append({"command": command, "kwargs": kwargs})
        if self.invalid_first and exec_count == 1:
            return subprocess.CompletedProcess(command, 0, "not-json", "first attempt invalid")
        packets = json.loads(kwargs["input"].split("\n\n", 1)[1])
        if self.omit_last:
            packets = packets[:-1]
        reviews = []
        for packet in packets:
            dimension = packet["audit_dimension"]
            reviews.append(
                {
                    "packet_id": packet["packet_id"],
                    "audit_dimension": dimension,
                    "verdict": "pass",
                    "rationale": "Packet evidence does not show a concern.",
                    "evidence": ["packet-only review"],
                }
            )
        return subprocess.CompletedProcess(command, 0, json.dumps({"reviews": reviews}), "")


class AutomatedAuditCodexTest(unittest.TestCase):
    def _inputs(self, root: Path) -> tuple[Path, Path, Path, Path]:
        output = root / "outputs"
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"output_dir": "outputs", "codex": {"model": "test-model"}}))
        construct = root / "construct.jsonl"
        construct.write_text(
            json.dumps({"packet_id": "construct-1", "case_id": "case-1", "claim": "x"}) + "\n"
            + json.dumps({"packet_id": "construct-2", "case_id": "case-2", "claim": "y"})
            + "\n"
        )
        temporal = root / "temporal.jsonl"
        temporal.write_text(json.dumps({"packet_id": "temporal-1", "case_id": "case-1", "prompt": "z"}) + "\n")
        return manifest, construct, temporal, output

    def test_runner_batches_records_provenance_and_uses_hardened_codex_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest, construct, temporal, output = self._inputs(root)
            fake = FakeCodex()
            report = run_codex_reviews(
                manifest_path=manifest,
                construct_packets_path=construct,
                temporal_packets_path=temporal,
                batch_size=2,
                workers=2,
                run_command=fake,
            )
            self.assertEqual(report["execution"]["packet_count"], 3)
            self.assertEqual(report["execution"]["batch_count"], 2)
            self.assertEqual(report["codex"]["version"], "codex-cli test-version")
            self.assertEqual(report["codex"]["model"], "test-model")
            self.assertEqual(report["inputs"]["construct_packets"]["sha256"], hashlib.sha256(construct.read_bytes()).hexdigest())
            reviews = [json.loads(line) for line in (output / REVIEWS_FILENAME).read_text().splitlines()]
            self.assertEqual({row["packet_id"] for row in reviews}, {"construct-1", "construct-2", "temporal-1"})
            self.assertTrue((output / RUN_FILENAME).is_file())
            for call in fake.calls:
                command = call["command"]
                self.assertIn("--ephemeral", command)
                self.assertIn("--ignore-user-config", command)
                self.assertIn("--ignore-rules", command)
                self.assertIn("read-only", command)
                self.assertIn("--output-schema", command)
                self.assertIn('approval_policy="never"', command)
                prompt_packets = json.loads(call["kwargs"]["input"].split("\n\n", 1)[1])
                self.assertGreaterEqual(call["kwargs"]["input"].count("packet_id"), len(prompt_packets))

    def test_runner_retries_invalid_structured_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest, construct, temporal, _ = self._inputs(root)
            fake = FakeCodex(invalid_first=True)
            report = run_codex_reviews(
                manifest_path=manifest,
                construct_packets_path=construct,
                temporal_packets_path=temporal,
                batch_size=10,
                retries=1,
                run_command=fake,
            )
            self.assertEqual(len(fake.calls), 2)
            self.assertEqual(len(report["batches"][0]["attempts"]), 2)
            self.assertEqual(report["batches"][0]["attempts"][0]["raw_last_message"], "not-json")

    def test_runner_rejects_duplicate_packets_and_incomplete_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest, construct, temporal, _ = self._inputs(root)
            temporal.write_text(json.dumps({"packet_id": "construct-1", "case_id": "case-1"}) + "\n")
            fake = FakeCodex()
            with self.assertRaisesRegex(AutomatedAuditError, "Duplicate packet_id"):
                run_codex_reviews(
                    manifest_path=manifest,
                    construct_packets_path=construct,
                    temporal_packets_path=temporal,
                    run_command=fake,
                )
            self.assertEqual(fake.calls, [])

            _, construct, temporal, output = self._inputs(root)
            with self.assertRaisesRegex(AutomatedAuditError, "omitted packet_id"):
                run_codex_reviews(
                    manifest_path=manifest,
                    construct_packets_path=construct,
                    temporal_packets_path=temporal,
                    retries=0,
                    run_command=FakeCodex(omit_last=True),
                )
            failure_report = json.loads((output / RUN_FILENAME).read_text())
            self.assertEqual(failure_report["status"], "failed")
            self.assertEqual(failure_report["execution"]["failed_batch_count"], 1)
            self.assertFalse(failure_report["reviews"]["partial_results_published"])
            self.assertFalse((output / REVIEWS_FILENAME).exists())

    def test_finalizer_applies_conservative_precedence_without_relabeling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest, _, _, output = self._inputs(root)
            deterministic = root / "deterministic.jsonl"
            rows = [
                {"case_id": "case-integrity", "integrity_errors": ["broken join"], "disposition": "include"},
                {"case_id": "case-label", "label_disagreement": True, "disposition": "include"},
                {"case_id": "case-construct", "disposition": "include"},
                {"case_id": "case-temporal", "disposition": "include"},
                {"case_id": "case-deterministic-temporal", "disposition": "include", "deterministic_temporal_leakage": True},
                {"case_id": "case-pass", "disposition": "include", "original_label": "TypeC"},
            ]
            deterministic.write_text("".join(json.dumps(row) + "\n" for row in rows))
            reviews = output / REVIEWS_FILENAME
            output.mkdir()
            review_rows = [
                self._review("p-integrity", "case-integrity", "construct", "concern"),
                self._review("p-label", "case-label", "construct", "pass"),
                self._review("p-construct", "case-construct", "construct", "uncertain"),
                self._review("p-temporal", "case-temporal", "temporal", "suspected_temporal_leakage"),
                self._review("p-pass", "case-pass", "construct", "pass"),
            ]
            reviews.write_text("".join(json.dumps(row) + "\n" for row in review_rows))
            report = finalize_audit(manifest_path=manifest, deterministic_dispositions_path=deterministic)
            dispositions = {
                row["case_id"]: row
                for row in (json.loads(line) for line in (output / DISPOSITIONS_FILENAME).read_text().splitlines())
            }
            self.assertEqual(dispositions["case-integrity"]["disposition"], "exclude")
            self.assertEqual(dispositions["case-label"]["disposition"], "diagnostic")
            self.assertEqual(dispositions["case-construct"]["disposition"], "diagnostic")
            self.assertEqual(dispositions["case-temporal"]["disposition"], "exclude_pending_rerender")
            self.assertEqual(
                dispositions["case-deterministic-temporal"]["disposition"], "exclude"
            )
            self.assertEqual(dispositions["case-pass"]["disposition"], "include")
            serialized = json.dumps(dispositions)
            self.assertNotIn("final_label", serialized)
            self.assertNotIn("EXTERNAL_CONFIRMED", serialized)
            self.assertEqual(report["counts"]["cases"], 6)
            for filename in (DISAGREEMENTS_FILENAME, DISPOSITIONS_FILENAME, FINAL_JSON_FILENAME, FINAL_MD_FILENAME):
                self.assertTrue((output / filename).is_file())

    def test_finalizer_rejects_duplicate_review_ids_and_relabel_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest, _, _, output = self._inputs(root)
            deterministic = root / "deterministic.jsonl"
            deterministic.write_text(json.dumps({"case_id": "case-1", "disposition": "include"}) + "\n")
            output.mkdir()
            review = self._review("packet-1", "case-1", "construct", "pass")
            (output / REVIEWS_FILENAME).write_text(json.dumps(review) + "\n" + json.dumps(review) + "\n")
            with self.assertRaisesRegex(AutomatedAuditError, "Duplicate packet_id"):
                finalize_audit(manifest_path=manifest, deterministic_dispositions_path=deterministic)

            review["proposed_label"] = "EXTERNAL_CONFIRMED"
            (output / REVIEWS_FILENAME).write_text(json.dumps(review) + "\n")
            with self.assertRaisesRegex(AutomatedAuditError, "missing or unsupported"):
                finalize_audit(manifest_path=manifest, deterministic_dispositions_path=deterministic)

            review.pop("proposed_label")
            review["rationale"] = "This would assert EXTERNAL_CONFIRMED."
            (output / REVIEWS_FILENAME).write_text(json.dumps(review) + "\n")
            with self.assertRaisesRegex(AutomatedAuditError, "must not relabel"):
                finalize_audit(manifest_path=manifest, deterministic_dispositions_path=deterministic)

    def test_manifest_bound_blinded_packet_contract_integrates_with_finalizer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "audit"
            output.mkdir()
            construct = output / "construct_review_packets.jsonl"
            temporal = output / "temporal_review_packets.jsonl"
            statuses = output / "deterministic_case_status.jsonl"
            private_map = output / "private_packet_map.json"
            construct.write_text(
                json.dumps(
                    {
                        "review_id": "construct_000001",
                        "review_type": "construct",
                        "blinded_case_id": "construct_000001",
                    }
                )
                + "\n"
            )
            temporal.write_text(
                json.dumps(
                    {
                        "review_id": "temporal_000001",
                        "review_type": "temporal",
                        "blinded_case_id": "temporal_000001",
                    }
                )
                + "\n"
            )
            statuses.write_text(
                json.dumps({"case_id": "Q1-case", "status": "pass", "finding_codes": []})
                + "\n"
                + json.dumps({"case_id": "Q2-case", "status": "disagreement", "finding_codes": ["x"]})
                + "\n"
            )
            private_map.write_text(
                json.dumps({"construct_000001": "Q1-case", "temporal_000001": "Q2-case"}) + "\n"
            )
            paths = {
                "construct_review_packets": construct,
                "temporal_review_packets": temporal,
                "deterministic_case_status": statuses,
                "private_packet_map": private_map,
            }
            manifest = output / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "output_dir": str(output),
                        "codex": {"model": "test-model"},
                        "artifacts": {
                            name: {
                                "path": path.name,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "size_bytes": path.stat().st_size,
                            }
                            for name, path in paths.items()
                        },
                    }
                )
                + "\n"
            )
            run_codex_reviews(manifest_path=manifest, shard_size=2, run_command=FakeCodex())
            report = finalize_audit(manifest_path=manifest)
            self.assertEqual(report["counts"]["cases"], 2)
            dispositions = {
                row["case_id"]: row["disposition"]
                for row in (
                    json.loads(line) for line in (output / DISPOSITIONS_FILENAME).read_text().splitlines()
                )
            }
            self.assertEqual(dispositions, {"Q1-case": "include", "Q2-case": "diagnostic"})

            review_lines = (output / REVIEWS_FILENAME).read_text().splitlines()
            (output / REVIEWS_FILENAME).write_text(review_lines[0] + "\n")
            with self.assertRaisesRegex(AutomatedAuditError, "packet coverage mismatch"):
                finalize_audit(manifest_path=manifest)

    @staticmethod
    def _review(packet_id: str, case_id: str, dimension: str, verdict: str) -> dict:
        return {
            "packet_id": packet_id,
            "case_id": case_id,
            "audit_dimension": dimension,
            "verdict": verdict,
            "rationale": "fixture rationale",
            "evidence": ["fixture evidence"],
        }


if __name__ == "__main__":
    unittest.main()
