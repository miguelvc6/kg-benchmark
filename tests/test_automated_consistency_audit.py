import csv
import json
import tempfile
import unittest
from pathlib import Path

from automated_consistency_audit import (
    WorldStateLookup,
    _case_findings,
    _changed_constraint_entries,
    _changed_constraint_types,
    _construct_packet,
    _local_support,
    run_audit,
)


def _abox_record(case_id: str = "repair_Q1_100001") -> dict:
    return {
        "id": case_id,
        "qid": "Q1",
        "property": "P1",
        "track": "A_BOX",
        "repair_target": {
            "kind": "A_BOX",
            "action": "UPDATE",
            "old_value": ["Q1"],
            "new_value": ["Q2"],
        },
        "classification": {"class": "TypeA", "subtype": "SELF_LINK_REJECTION"},
        "context_ref": {"world_state_id": case_id},
    }


def _abox_world() -> dict:
    return {
        "L1_ego_node": {"properties": {"P1": ["Q2"], "P9": ["Q8"]}},
        "L2_labels": {"entities": {}},
        "L3_neighborhood": {"outgoing_edges": []},
        "L4_constraints": {"property_id": "P1", "constraints": []},
    }


class AutomatedConsistencyAuditTests(unittest.TestCase):
    def test_world_state_lookup_reads_canonical_jsonl_starting_with_object_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            world = root / "world-state.jsonl"
            rows = [
                {"id": "case-1", "world_state": _abox_world()},
                {"id": "case-2", "world_state": {**_abox_world(), "marker": 2}},
            ]
            world.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            with WorldStateLookup(world, root / "cache") as lookup:
                self.assertEqual(lookup.get("case-1"), _abox_world())
                self.assertEqual(lookup.get("case-2")["marker"], 2)

    def test_independent_locus_and_type_a_replay_pass(self) -> None:
        self.assertEqual(_case_findings(_abox_record(), _abox_world()), [])

    def test_locus_disagreement_is_reported(self) -> None:
        record = _abox_record()
        record["track"] = "T_BOX"
        findings = _case_findings(record, _abox_world())
        self.assertIn("locus_disagreement", {finding["code"] for finding in findings})

    def test_tbox_changed_types_use_presence_not_qualifier_changes(self) -> None:
        before = {"signature": [{"constraint_qid": "Q1", "qualifiers": [{"values": ["old"]}]}]}
        after = {"signature": [{"constraint_qid": "Q1", "qualifiers": [{"values": ["new"]}]}]}
        self.assertEqual(_changed_constraint_types(before, after), set())
        self.assertEqual(_changed_constraint_entries(before, after), {"Q1"})

    def test_tbox_duplicate_constraint_entries_preserve_qualifier_changes(self) -> None:
        before = {
            "signature": [
                {"constraint_qid": "Q1", "qualifiers": [{"values": ["changed-old"]}]},
                {"constraint_qid": "Q1", "qualifiers": [{"values": ["same"]}]},
            ]
        }
        after = {
            "signature": [
                {"constraint_qid": "Q1", "qualifiers": [{"values": ["changed-new"]}]},
                {"constraint_qid": "Q1", "qualifiers": [{"values": ["same"]}]},
            ]
        }
        self.assertEqual(_changed_constraint_entries(before, after), {"Q1"})

    def test_target_only_l2_label_is_not_counted_as_independent_local_support(self) -> None:
        record = _abox_record()
        record["classification"] = {"class": "TypeC", "subtype": "EXTERNAL_BY_ELIMINATION"}
        world = _abox_world()
        world["L2_labels"] = {"entities": {"Q2": {"label": "Target label"}}}
        self.assertEqual(_local_support(record, world), (False, []))

    def test_cardinality_and_format_contradictions_are_deterministic(self) -> None:
        record = _abox_record()
        record["repair_target"]["new_value"] = ["ABC", "DEF"]
        record["classification"] = {"class": "TypeB", "subtype": "LOCAL_TEXT_CONFIRMED"}
        world = _abox_world()
        world["L1_ego_node"]["properties"]["P9"] = ["ABC"]
        world["L4_constraints"]["constraints"] = [
            {"constraint_type": {"qid": "Q19474404"}, "qualifiers": []},
            {"constraint_type": {"qid": "Q21502404"}, "qualifiers": [{"property_id": "P1793", "values": [{"raw": "[A-Z]+"}]}]},
            {"constraint_type": {"qid": "Q21502404"}, "qualifiers": [{"property_id": "P1793", "values": [{"raw": "[0-9]+"}]}]},
        ]
        codes = {finding["code"] for finding in _case_findings(record, world)}
        self.assertIn("cardinality_constraint_conflict", codes)
        self.assertIn("format_rule_contradiction", codes)

    def test_construct_packet_omits_gold_labels_and_target_from_locus_view(self) -> None:
        packet = _construct_packet("repair_Q1_100001", "construct_000001", _abox_record(), _abox_world())
        rendered = json.dumps(packet)
        self.assertNotIn("classification", rendered)
        self.assertNotIn('"track"', rendered)
        self.assertNotIn('"kind"', rendered)
        self.assertNotIn("repair_Q1_100001", rendered)
        self.assertNotIn("repair_target", json.dumps(packet["locus_view"]))

    def test_end_to_end_sample_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage4 = root / "stage4.jsonl"
            world = root / "world.json"
            schema = root / "schema.json"
            sample = root / "sample.csv"
            prompts = root / "prompts.jsonl"
            render_summary = root / "render_summary.json"
            output = root / "output"
            record = _abox_record()
            stage4.write_text(json.dumps(record) + "\n", encoding="utf-8")
            world.write_text(json.dumps({record["id"]: _abox_world()}), encoding="utf-8")
            schema.write_text(
                json.dumps(
                    {
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "required": ["id", "track", "repair_target", "classification", "context_ref"],
                    }
                ),
                encoding="utf-8",
            )
            with sample.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["case_id", "classification"])
                writer.writeheader()
                writer.writerow({"case_id": record["id"], "classification": "hidden input must be ignored"})
            prompts.write_text(
                json.dumps(
                    {
                        "matrix_id": "m1",
                        "case_id": record["id"],
                        "task": "a_box_repair",
                        "context_bundle": "logic_only",
                        "historical_track": "A_BOX",
                        "system_prompt": "Use only the visible evidence.",
                        "user_prompt": "Remove the invalid self-link.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            render_summary.write_text(json.dumps({"rendered_prompt_count": 1}), encoding="utf-8")

            manifest = run_audit(
                classified_benchmark_path=stage4,
                world_state_path=world,
                stage4_schema_path=schema,
                construct_sample_path=sample,
                rendered_prompts_path=prompts,
                render_summary_path=render_summary,
                output_dir=output,
                construct_review_size=1,
                temporal_review_size=1,
                cache_dir=root / "cache",
            )

            self.assertEqual(manifest["counts"]["stage4_rows"], 1)
            self.assertEqual(manifest["counts"]["construct_review_packets"], 1)
            self.assertEqual(manifest["counts"]["temporal_review_packets"], 1)
            summary = json.loads((output / "deterministic_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status_counts"], {"pass": 1})
            self.assertEqual(summary["coverage"]["stage2_reconstruction"], "unavailable")
            packet = json.loads((output / "construct_review_packets.jsonl").read_text(encoding="utf-8"))
            self.assertNotIn("hidden input must be ignored", json.dumps(packet))

            with self.assertRaisesRegex(ValueError, "not empty"):
                run_audit(
                    classified_benchmark_path=stage4,
                    world_state_path=world,
                    stage4_schema_path=schema,
                    construct_sample_path=sample,
                    rendered_prompts_path=prompts,
                    render_summary_path=render_summary,
                    output_dir=output,
                    construct_review_size=1,
                    temporal_review_size=1,
                    cache_dir=root / "cache",
                )


if __name__ == "__main__":
    unittest.main()
