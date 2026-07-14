from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from kg_benchmark.selection.workflow import (
    SelectionWorkflowError,
    expand_population,
    finalize_selection,
    prepare_reserve,
    review_reserve,
    selection_status,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeCodex:
    def __init__(self, fail_packet_id: str) -> None:
        self.fail_packet_id = fail_packet_id

    def __call__(self, command, **kwargs):
        if command == ["codex", "--version"]:
            return subprocess.CompletedProcess(command, 0, "codex-cli selection-test-1.0\n", "")
        packets = json.loads(kwargs["input"].split("\n\n", 1)[1])
        reviews = []
        for packet in packets:
            reviews.append(
                {
                    "packet_id": packet["packet_id"],
                    "audit_dimension": "temporal",
                    "verdict": "uncertain" if packet["packet_id"] == self.fail_packet_id else "pass",
                    "rationale": "Synthetic blinded temporal inspection.",
                    "evidence": ["synthetic reserve prompt"],
                }
            )
        return subprocess.CompletedProcess(command, 0, json.dumps({"reviews": reviews}), "")


def _abox(index: int, class_name: str, subtype: str) -> tuple[dict, dict]:
    case_id = f"abox-case-{index:04d}"
    qid = f"Q{10000 + index}"
    pid = f"P{20000 + index}"
    old_value = f"Q{30000 + index}"
    new_value = f"Q{40000 + index}"
    record = {
        "id": case_id,
        "qid": qid,
        "property": pid,
        "track": "A_BOX",
        "labels_en": {"qid": f"Entity {index}", "property": f"Property {index}"},
        "violation_context": {"value": [old_value]},
        "repair_target": {
            "kind": "A_BOX",
            "action": "UPDATE",
            "old_value": [old_value],
            "new_value": [new_value],
        },
        "classification": {"class": class_name, "subtype": subtype},
        "context_ref": {"world_state_id": case_id},
    }
    world = {
        "L1_ego_node": {"qid": qid, "properties": {pid: [new_value], "P999": [f"Q{50000 + index}"]}},
        "L2_labels": {"entities": {}},
        "L3_neighborhood": {"outgoing_edges": []},
        "L4_constraints": {"property_id": pid, "constraints": []},
    }
    return record, world


def _tbox(index: int, category: str, role_index: int) -> tuple[dict, dict]:
    case_id = f"tbox-case-{index:04d}"
    qid = f"Q{60000 + index}"
    pid = f"P{70000 + index}"
    subtype = {
        "relaxation_expansions": "RELAXATION_SET_EXPANSION",
        "restriction_contractions": "RESTRICTION_SET_CONTRACTION",
        "schema_updates": "SCHEMA_UPDATE",
    }[category]
    role_gold = (
        {
            "schema_decision": "CAUSAL_SCHEMA_REPAIR",
            "repairs": [{"repair_op": "CONSTRAINT_QUALIFIER_ADD", "taxonomy_code": "CQ_PLUS"}],
        },
        {
            "schema_decision": "CAUSAL_SCHEMA_REPAIR",
            "repairs": [{"repair_op": "CONSTRAINT_QUALIFIER_REMOVE", "taxonomy_code": "CQ_MINUS"}],
        },
        {"schema_decision": "NO_CAUSAL_SCHEMA_REPAIR", "repairs": []},
        {
            "schema_decision": "UNCLEAR_SCHEMA_EVIDENCE",
            "repairs": [{"repair_op": "OTHER", "taxonomy_code": "OTHER"}],
        },
    )[role_index % 4]
    gold = {
        "case_id": case_id,
        "target": {"pid": pid, "constraint_type_qid": "Q21510859"},
        **role_gold,
    }
    record = {
        "id": case_id,
        "qid": qid,
        "property": pid,
        "track": "T_BOX",
        "labels_en": {"qid": f"Entity {index}", "property": f"Property {index}"},
        "repair_target": {"property_revision_id": f"revision-{90000 + index}"},
        "classification": {"class": "T_BOX", "subtype": subtype},
        "context_ref": {"world_state_id": case_id},
        "gold": gold,
    }
    world = {
        "L1_ego_node": {"qid": qid, "properties": {"P999": [f"Q{80000 + index}"]}},
        "L2_labels": {"entities": {}},
        "L3_neighborhood": {"outgoing_edges": []},
        "L4_constraints": {"property_id": pid, "constraints": []},
    }
    return record, world


class SelectionWorkflowTests(unittest.TestCase):
    def _inputs(self, root: Path) -> dict[str, Path]:
        records: list[dict] = []
        worlds: dict[str, dict] = {}
        index = 1
        for offset in range(36):
            subtype = "TARGET_REQUIRED_CLAIM" if offset % 2 == 0 else "FORMAT_NORMALIZATION"
            record, world = _abox(index, "TypeA", subtype)
            records.append(record)
            worlds[record["id"]] = world
            index += 1
        for _ in range(30):
            record, world = _abox(index, "TypeB", "LOCAL_TEXT_CONFIRMED")
            records.append(record)
            worlds[record["id"]] = world
            index += 1
        for _ in range(30):
            record, world = _abox(index, "TypeC", "EXTERNAL_BY_ELIMINATION")
            records.append(record)
            worlds[record["id"]] = world
            index += 1
        categories = ["relaxation_expansions"] * 18 + ["restriction_contractions"] * 12 + ["schema_updates"] * 24
        for role_index, category in enumerate(categories):
            record, world = _tbox(index, category, role_index)
            records.append(record)
            worlds[record["id"]] = world
            index += 1

        cases = root / "cases.jsonl"
        world_state = root / "world.json"
        dispositions = root / "dispositions.jsonl"
        cases.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        world_state.write_text(json.dumps(worlds), encoding="utf-8")
        dispositions.write_text(
            "".join(json.dumps({"case_id": row["id"], "disposition": "include"}) + "\n" for row in records),
            encoding="utf-8",
        )
        audit_summary = root / "audit-summary.json"
        audit_summary.write_text(
            json.dumps(
                {
                    "validation": {
                        "complete_unique_disposition_coverage": True,
                        "deterministic_gates_passed": True,
                        "temporal_gate_passed": True,
                        "review_complete": True,
                        "unresolved_systemic_findings": 0,
                        "selection_eligible_disposition": "include",
                    },
                    "provenance": {
                        "cases": {"sha256": hashlib.sha256(cases.read_bytes()).hexdigest()},
                        "dispositions": {"sha256": hashlib.sha256(dispositions.read_bytes()).hexdigest()},
                    },
                }
            ),
            encoding="utf-8",
        )
        exclusions = root / "exclusions.json"
        excluded = records[0]
        exclusions.write_text(
            json.dumps(
                {
                    "manifest_type": "event_group_exclusions",
                    "manifest_version": 1,
                    "description": "Synthetic prior paper group.",
                    "group_keys": [f"ABOX|{excluded['qid']}|{excluded['property']}"],
                }
            ),
            encoding="utf-8",
        )
        policy = root / "selection-policy.json"
        policy_payload = json.loads((ROOT / "paper" / "selection-policy.json").read_text(encoding="utf-8"))
        policy_payload["populations"] = {
            "reserve-1440": {"IC-L": 20, "IC-G": 20, "IC-E-elim": 20, "TBOX": 20},
            "main-1200": {"IC-L": 10, "IC-G": 10, "IC-E-elim": 10, "TBOX": 10},
            "azure-600": {"IC-L": 5, "IC-G": 5, "IC-E-elim": 5, "TBOX": 5},
        }
        policy_payload["tbox_main_target"] = {
            "relaxation_expansions": 4,
            "restriction_contractions": 2,
            "schema_updates": 4,
            "allocation_role": "approximate_target_subject_to_eligible_prefix",
        }
        policy.write_text(json.dumps(policy_payload), encoding="utf-8")
        protocol = root / "protocol.json"
        protocol.write_text((ROOT / "paper" / "protocol.json").read_text(encoding="utf-8"), encoding="utf-8")
        return {
            "cases": cases,
            "world": world_state,
            "dispositions": dispositions,
            "audit": audit_summary,
            "exclusions": exclusions,
            "policy": policy,
            "protocol": protocol,
        }

    def test_reserve_review_replacement_finalization_and_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = self._inputs(root)
            output = root / "selection"
            state = prepare_reserve(
                cases_path=inputs["cases"],
                world_state_path=inputs["world"],
                dispositions_path=inputs["dispositions"],
                audit_summary_path=inputs["audit"],
                exclusions_path=inputs["exclusions"],
                protocol_path=inputs["protocol"],
                policy_path=inputs["policy"],
                output_dir=output,
                repo_root=ROOT,
            )
            self.assertEqual(state["phases"]["reserve"]["counts"]["reserve_cases"], 80)
            self.assertEqual(state["phases"]["reserve"]["counts"]["rendered_prompts"], 640)
            self.assertEqual(state["phases"]["reserve"]["counts"]["temporal_review_cases"], 50)
            self.assertEqual(
                prepare_reserve(
                    cases_path=inputs["cases"],
                    world_state_path=inputs["world"],
                    dispositions_path=inputs["dispositions"],
                    audit_summary_path=inputs["audit"],
                    exclusions_path=inputs["exclusions"],
                    protocol_path=inputs["protocol"],
                    policy_path=inputs["policy"],
                    output_dir=output,
                    repo_root=ROOT,
                ),
                state,
            )
            changed_exclusions = root / "changed-exclusions.json"
            changed_exclusions.write_text(
                json.dumps(
                    {
                        "manifest_type": "event_group_exclusions",
                        "manifest_version": 1,
                        "group_keys": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SelectionWorkflowError, "different exclusions"):
                prepare_reserve(
                    cases_path=inputs["cases"],
                    world_state_path=inputs["world"],
                    dispositions_path=inputs["dispositions"],
                    audit_summary_path=inputs["audit"],
                    exclusions_path=changed_exclusions,
                    protocol_path=inputs["protocol"],
                    policy_path=inputs["policy"],
                    output_dir=output,
                    repo_root=ROOT,
                )

            reserve = json.loads((output / "reserve.json").read_text(encoding="utf-8"))
            initial_main: set[str] = set()
            offset = 0
            for stratum in ("IC-L", "IC-G", "IC-E-elim", "TBOX"):
                count = reserve["quotas"][stratum]
                initial_main.update(reserve["selected_case_ids"][offset : offset + 10])
                offset += count
            private_map = json.loads((output / "private-temporal-review-map.json").read_text(encoding="utf-8"))
            fail_packet_id = next(packet_id for packet_id, case_id in private_map.items() if case_id in initial_main)
            review_reserve(
                output_dir=output,
                repo_root=ROOT,
                batch_size=10,
                workers=1,
                run_command=FakeCodex(fail_packet_id),
            )
            state = finalize_selection(output_dir=output, repo_root=ROOT)
            self.assertEqual(state["phases"]["finalize"]["counts"]["main_cases"], 40)
            self.assertEqual(state["phases"]["finalize"]["counts"]["azure_cases"], 20)
            self.assertGreaterEqual(state["phases"]["finalize"]["counts"]["replacements"], 1)

            main = json.loads((output / "main-1200.json").read_text(encoding="utf-8"))
            azure = json.loads((output / "azure-600.json").read_text(encoding="utf-8"))
            self.assertEqual(main["tbox_composition"], {
                "relaxation_expansions": 4,
                "restriction_contractions": 2,
                "schema_updates": 4,
            })
            self.assertTrue(set(azure["selected_case_ids"]).issubset(main["selected_case_ids"]))
            self.assertNotIn(private_map[fail_packet_id], main["selected_case_ids"])
            self.assertEqual(set(main["case_eligibility_sha256"]), set(main["selected_case_ids"]))
            self.assertEqual(set(azure["case_eligibility_sha256"]), set(azure["selected_case_ids"]))
            self.assertTrue(
                all(
                    re.fullmatch(r"[0-9a-f]{64}", digest)
                    for digest in main["case_eligibility_sha256"].values()
                )
            )
            excluded_group = json.loads(inputs["exclusions"].read_text())["group_keys"][0]
            self.assertNotIn(excluded_group, main["selected_group_keys"])
            support = json.loads((output / "support-bank.json").read_text(encoding="utf-8"))
            support_groups = {row["group_key"] for rows in support["support_sets"].values() for row in rows}
            self.assertTrue(support_groups.isdisjoint(main["selected_group_keys"]))
            for role in (
                "dataset",
                "audit",
                "exclusions",
                "ranking",
                "support_bank",
                "eligibility_order",
                "prompt_audit",
                "per_case_eligibility",
            ):
                self.assertRegex(main["provenance"][role]["sha256"], r"^[0-9a-f]{64}$")

            extension_path = output / "main-expanded.json"
            expanded = expand_population(
                output_dir=output,
                parent_path=output / "main-1200.json",
                name="main-expanded",
                requested_quotas={"IC-L": 12, "IC-G": 12, "IC-E-elim": 12, "TBOX": 25},
                destination=extension_path,
                repo_root=ROOT,
            )
            self.assertEqual(expanded["case_count"], 61)
            self.assertLess(expanded["quotas"]["TBOX"], 25)
            self.assertEqual(sum(expanded["quotas"].values()), 61)
            self.assertTrue(set(main["selected_case_ids"]).issubset(expanded["selected_case_ids"]))
            self.assertEqual(selection_status(output_dir=output, repo_root=ROOT)["valid_through"], "finalize")

            selected_id = main["selected_case_ids"][0]
            main["case_eligibility_sha256"][selected_id] = "0" * 64
            (output / "main-1200.json").write_text(
                json.dumps(main, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            workflow_path = output / "selection-workflow.json"
            workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
            workflow["phases"]["finalize"]["artifacts"]["main"]["sha256"] = hashlib.sha256(
                (output / "main-1200.json").read_bytes()
            ).hexdigest()
            workflow["phases"]["finalize"]["artifacts"]["main"]["size_bytes"] = (
                output / "main-1200.json"
            ).stat().st_size
            workflow_path.write_text(
                json.dumps(workflow, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SelectionWorkflowError, "mismatched eligibility digest"):
                selection_status(output_dir=output, repo_root=ROOT)


if __name__ == "__main__":
    unittest.main()
