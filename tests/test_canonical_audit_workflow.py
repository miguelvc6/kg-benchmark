from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from kg_benchmark.audit.workflow import (
    AuditWorkflowError,
    audit_status,
    prepare_audit,
    run_deterministic_phase,
    run_finalize_phase,
    run_review_phase,
)
from kg_benchmark.cli import _main

ROOT = Path(__file__).resolve().parents[1]


class FakeCodex:
    def __call__(self, command, **kwargs):
        if command == ["codex", "--version"]:
            return subprocess.CompletedProcess(command, 0, "codex-cli synthetic-1.0\n", "")
        packets = json.loads(kwargs["input"].split("\n\n", 1)[1])
        reviews = [
            {
                "packet_id": packet["packet_id"],
                "audit_dimension": packet["audit_dimension"],
                "verdict": "pass",
                "rationale": "The blinded packet contains no construct or temporal concern.",
                "evidence": ["synthetic packet inspection"],
            }
            for packet in packets
        ]
        return subprocess.CompletedProcess(command, 0, json.dumps({"reviews": reviews}), "")


def _record(case_id: str, qid: str, old_value: str, new_value: str) -> dict:
    return {
        "id": case_id,
        "qid": qid,
        "property": "P1",
        "track": "A_BOX",
        "repair_target": {
            "kind": "A_BOX",
            "action": "UPDATE",
            "old_value": [old_value],
            "new_value": [new_value],
        },
        "classification": {"class": "TypeA", "subtype": "SELF_LINK_REJECTION"},
        "context_ref": {"world_state_id": case_id},
    }


def _world(new_value: str) -> dict:
    return {
        "L1_ego_node": {"properties": {"P1": [new_value], "P9": ["Q90"]}},
        "L2_labels": {"entities": {}},
        "L3_neighborhood": {"outgoing_edges": []},
        "L4_constraints": {"property_id": "P1", "constraints": []},
    }


class CanonicalAuditWorkflowTests(unittest.TestCase):
    def _inputs(self, root: Path) -> dict[str, Path]:
        cases = root / "cases.jsonl"
        records = [
            _record("repair_Q1_100001", "Q1", "Q1", "Q2"),
            _record("repair_Q3_100002", "Q3", "Q3", "Q4"),
        ]
        cases.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        world = root / "world.json"
        world.write_text(
            json.dumps({records[0]["id"]: _world("Q2"), records[1]["id"]: _world("Q4")}),
            encoding="utf-8",
        )
        schema = root / "case.schema.json"
        schema.write_text(
            json.dumps(
                {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "required": ["id", "qid", "property", "track", "repair_target", "classification", "context_ref"],
                }
            ),
            encoding="utf-8",
        )
        protocol = root / "protocol.json"
        payload = json.loads((ROOT / "paper" / "protocol.json").read_text(encoding="utf-8"))
        payload["audit"]["construct_review"]["sample_size"] = 2
        payload["audit"]["construct_review"]["reviewer_model"] = "synthetic-reviewer"
        payload["audit"]["temporal_review"]["sample_size"] = 2
        protocol.write_text(json.dumps(payload), encoding="utf-8")
        return {"cases": cases, "world": world, "schema": schema, "protocol": protocol}

    def _prepare(self, root: Path) -> tuple[dict[str, Path], Path]:
        inputs = self._inputs(root)
        work = root / "audit"
        prepare_audit(
            cases_path=inputs["cases"],
            world_state_path=inputs["world"],
            stage4_schema_path=inputs["schema"],
            protocol_path=inputs["protocol"],
            work_dir=work,
            repo_root=ROOT,
        )
        return inputs, work

    def test_public_prepare_generates_label_hidden_deterministic_sample_and_all_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = self._inputs(root)
            stage2 = root / "stage2-placeholder.jsonl"
            stage2.write_text("{}\n", encoding="utf-8")
            lineage = root / "lineage-placeholder.json"
            lineage.write_text("{}\n", encoding="utf-8")
            work = root / "public-audit"
            result = _main(
                [
                    "audit",
                    "prepare",
                    "--cases",
                    str(inputs["cases"]),
                    "--world-state",
                    str(inputs["world"]),
                    "--stage2",
                    str(stage2),
                    "--lineage-manifest",
                    str(lineage),
                    "--stage4-schema",
                    str(inputs["schema"]),
                    "--protocol",
                    str(inputs["protocol"]),
                    "--work-dir",
                    str(work),
                ]
            )
            self.assertEqual(result, 0)
            with (work / "construct-sample.csv").open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, ["case_id"])
            self.assertEqual(len(rows), 2)
            expected = sorted(
                ["repair_Q1_100001", "repair_Q3_100002"],
                key=lambda case_id: hashlib.sha256(f"13|construct_review|{case_id}".encode()).hexdigest(),
            )
            self.assertEqual([row["case_id"] for row in rows], expected)
            prompts = [json.loads(line) for line in (work / "rendered-prompts.jsonl").read_text().splitlines()]
            self.assertEqual(len(prompts), 8)
            self.assertEqual({row["task"] for row in prompts}, {"a_box_repair", "track_diagnosis"})
            self.assertEqual({row["context_bundle"] for row in prompts}, {"logic_only", "local_graph"})
            visible_text = "\n".join(row["system_prompt"] + row["user_prompt"] for row in prompts)
            self.assertNotIn("repair_Q1_100001", visible_text)
            self.assertNotIn("repair_Q3_100002", visible_text)

    def test_full_workflow_is_resumable_hash_bound_and_writes_canonical_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, work = self._prepare(root)
            prepared_bytes = (work / "workflow.json").read_bytes()
            # A completed phase is reused rather than rewritten.
            state = prepare_audit(
                cases_path=root / "cases.jsonl",
                world_state_path=root / "world.json",
                stage4_schema_path=root / "case.schema.json",
                protocol_path=root / "protocol.json",
                work_dir=work,
                repo_root=ROOT,
            )
            self.assertEqual((work / "workflow.json").read_bytes(), prepared_bytes)
            self.assertEqual(state["phases"]["prepare"]["counts"]["rendered_prompts"], 8)

            state = run_deterministic_phase(work_dir=work, repo_root=ROOT, cache_dir=root / "cache")
            self.assertTrue(state["phases"]["deterministic"]["passed"])
            temporal = json.loads((work / "deterministic" / "temporal_audit.json").read_text(encoding="utf-8"))
            sampled_case_ids = [row["case_id"] for row in temporal["manual_review_sample"]]
            self.assertEqual(len(sampled_case_ids), len(set(sampled_case_ids)))
            state = run_review_phase(
                work_dir=work,
                repo_root=ROOT,
                batch_size=2,
                workers=1,
                run_command=FakeCodex(),
            )
            self.assertEqual(state["phases"]["review"]["reviewer"]["model"], "synthetic-reviewer")
            report = root / "audit.md"
            state = run_finalize_phase(work_dir=work, report_path=report, repo_root=ROOT)
            self.assertEqual(state["phases"]["finalize"]["status"], "complete")

            dispositions = [json.loads(line) for line in (work / "dispositions.jsonl").read_text().splitlines()]
            self.assertEqual(len(dispositions), 2)
            self.assertTrue(all(row["disposition"] == "include" for row in dispositions))
            summary = json.loads((work / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["reviewer"]["cli_version"], "codex-cli synthetic-1.0")
            self.assertEqual(summary["counts"]["construct_reviews"], 2)
            self.assertEqual(summary["counts"]["temporal_reviews"], 2)
            self.assertEqual(summary["counts"]["rendered_prompts"], 8)
            self.assertEqual(summary["provenance"]["dispositions"]["sha256"], hashlib.sha256((work / "dispositions.jsonl").read_bytes()).hexdigest())
            for role in (
                "protocol",
                "cases",
                "world_state",
                "stage4_schema",
                "construct_sample",
                "rendered_prompts",
                "render_summary",
                "deterministic_manifest",
                "review_schema",
                "review_run",
                "reviews",
                "dispositions",
            ):
                self.assertRegex(summary["provenance"][role]["sha256"], r"^[0-9a-f]{64}$")
            self.assertIn("# Dataset Audit", report.read_text(encoding="utf-8"))
            self.assertEqual(audit_status(work_dir=work, repo_root=ROOT)["valid_through"], "finalize")

            with (work / "dispositions.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(AuditWorkflowError, "changed after"):
                audit_status(work_dir=work, repo_root=ROOT)

    def test_completed_prepare_rejects_tampered_sample_before_deterministic_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, work = self._prepare(root)
            with (work / "construct-sample.csv").open("a", encoding="utf-8") as handle:
                handle.write("unbound-case\n")
            with self.assertRaisesRegex(AuditWorkflowError, "changed after"):
                run_deterministic_phase(work_dir=work, repo_root=ROOT)

    def test_deterministic_temporal_hit_is_permanently_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = self._inputs(root)
            cases = [json.loads(line) for line in inputs["cases"].read_text(encoding="utf-8").splitlines()]
            cases[0]["repair_target"]["author"] = "HiddenEditor"
            inputs["cases"].write_text(
                "".join(json.dumps(row) + "\n" for row in cases), encoding="utf-8"
            )
            world = json.loads(inputs["world"].read_text(encoding="utf-8"))
            world["repair_Q1_100001"]["L1_ego_node"]["properties"]["P9"] = ["HiddenEditor"]
            inputs["world"].write_text(json.dumps(world), encoding="utf-8")
            work = root / "audit"
            prepare_audit(
                cases_path=inputs["cases"],
                world_state_path=inputs["world"],
                stage4_schema_path=inputs["schema"],
                protocol_path=inputs["protocol"],
                work_dir=work,
                repo_root=ROOT,
            )

            state = run_deterministic_phase(work_dir=work, repo_root=ROOT, cache_dir=root / "cache")
            self.assertTrue(state["phases"]["deterministic"]["passed"])
            temporal = json.loads(
                (work / "deterministic" / "temporal_audit.json").read_text(encoding="utf-8")
            )
            self.assertFalse(temporal["passed_automated_gate"])
            self.assertTrue(temporal["passed_case_exclusion_gate"])
            self.assertEqual(temporal["excluded_case_ids"], ["repair_Q1_100001"])
            statuses = [
                json.loads(line)
                for line in (work / "deterministic" / "deterministic_case_status.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            leaked = next(row for row in statuses if row["case_id"] == "repair_Q1_100001")
            self.assertTrue(leaked["deterministic_temporal_leakage"])

            run_review_phase(
                work_dir=work,
                repo_root=ROOT,
                batch_size=2,
                workers=1,
                run_command=FakeCodex(),
            )
            run_finalize_phase(work_dir=work, report_path=root / "audit.md", repo_root=ROOT)
            dispositions = {
                row["case_id"]: row["disposition"]
                for row in (
                    json.loads(line)
                    for line in (work / "dispositions.jsonl").read_text(encoding="utf-8").splitlines()
                )
            }
            self.assertEqual(dispositions["repair_Q1_100001"], "exclude")
            self.assertEqual(dispositions["repair_Q3_100002"], "include")


if __name__ == "__main__":
    unittest.main()
