import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rescore_run import rescore_run


class RescoreRunTests(unittest.TestCase):
    def test_replays_each_bundle_without_model_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            benchmark = root / "classified.jsonl"
            world = root / "world.json"
            selection = root / "selection.json"
            benchmark.write_text('{"id":"case-1"}\n', encoding="utf-8")
            world.write_text("{}\n", encoding="utf-8")
            selection.write_text('{"selected_case_ids":["case-1"]}\n', encoding="utf-8")
            (run_dir / "run_manifest.jsonl").write_text("", encoding="utf-8")
            (run_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "classified_benchmark": str(benchmark),
                        "world_state": str(world),
                        "selection_manifest": str(selection),
                        "selected_case_ids": ["case-1"],
                        "ablation_bundles": ["logic_only", "local_graph"],
                        "tbox_task_version": "strict_signature_after_v1",
                    }
                ),
                encoding="utf-8",
            )
            for bundle in ("logic_only", "local_graph"):
                bundle_dir = run_dir / bundle
                bundle_dir.mkdir()
                (bundle_dir / "a_box_proposals.jsonl").write_text("", encoding="utf-8")
                (bundle_dir / "t_box_proposals.jsonl").write_text("", encoding="utf-8")
                (bundle_dir / "track_diagnoses.jsonl").write_text("", encoding="utf-8")

            calls = []

            def fake_evaluate_benchmark(**kwargs):
                calls.append(kwargs)
                Path(kwargs["out_traces_path"]).write_text("", encoding="utf-8")
                summary = {"bundle": kwargs["ablation_bundle"], "metric_version": "v2"}
                Path(kwargs["out_summary_path"]).write_text(json.dumps(summary), encoding="utf-8")
                return [], summary

            with patch("rescore_run.evaluate_benchmark", side_effect=fake_evaluate_benchmark):
                manifest = rescore_run(run_dir=run_dir, evaluation_id="metrics_v2")

            self.assertEqual(len(calls), 2)
            self.assertEqual(manifest["provider_calls"], 0)
            self.assertEqual(manifest["selected_case_count"], 1)
            self.assertEqual(manifest["ablation_bundles"], ["logic_only", "local_graph"])
            self.assertTrue((run_dir / "evaluations" / "metrics_v2" / "evaluation_manifest.json").is_file())
            with self.assertRaises(FileExistsError):
                rescore_run(run_dir=run_dir, evaluation_id="metrics_v2")

    def test_replays_taxonomy_patch_metrics_without_strict_tbox_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            bundle_dir = run_dir / "logic_only"
            bundle_dir.mkdir(parents=True)
            benchmark = root / "classified.jsonl"
            world = root / "world.json"
            selection = root / "selection.json"
            benchmark.write_text(
                json.dumps(
                    {
                        "id": "tbox-1",
                        "qid": "Q1",
                        "property": "P31",
                        "track": "T_BOX",
                        "repair_target": {"kind": "T_BOX", "property_revision_id": 1},
                        "classification": {
                            "class": "T_BOX",
                            "subtype": "SCHEMA_UPDATE",
                            "confidence": "high",
                            "diagnostics": {
                                "tbox_diff_summary": {"target_constraint_qid": "Q21510859"}
                            },
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            world.write_text("{}\n", encoding="utf-8")
            selection.write_text('{"selected_case_ids":["tbox-1"]}\n', encoding="utf-8")
            (run_dir / "run_manifest.jsonl").write_text("", encoding="utf-8")
            (run_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "classified_benchmark": str(benchmark),
                        "world_state": str(world),
                        "selection_manifest": str(selection),
                        "selected_case_ids": ["tbox-1"],
                        "ablation_bundles": ["logic_only"],
                        "tbox_task_version": "tbox_taxonomy_patch_v1",
                    }
                ),
                encoding="utf-8",
            )
            for filename in (
                "a_box_proposals.jsonl",
                "t_box_proposals.jsonl",
                "track_diagnoses.jsonl",
            ):
                (bundle_dir / filename).write_text("", encoding="utf-8")
            prediction = {
                "case_id": "tbox-1",
                "schema_decision": "CAUSAL_SCHEMA_REPAIR",
                "target": {"pid": "P31", "constraint_type_qid": "Q21510859"},
                "repairs": [
                    {
                        "repair_op": "OTHER_TBOX_UPDATE",
                        "taxonomy_code": "OTHER",
                        "constraint_type_qid": "Q21510859",
                        "qualifier_property_id": None,
                        "added_values": [],
                        "removed_values": [],
                        "old_value": None,
                        "new_value": None,
                        "rank_after": None,
                        "snaktype_after": None,
                        "evidence_level": "FAMILY_ONLY",
                    }
                ],
                "rationale": "Visible schema family changed.",
                "provenance": [{"kind": "KG", "node_id": "P31", "snippet": "visible"}],
                "uncertainty": {"confidence": 0.8, "notes": "family only"},
            }
            (bundle_dir / "t_box_taxonomy_patch_proposals.jsonl").write_text(
                json.dumps(prediction) + "\n", encoding="utf-8"
            )

            manifest = rescore_run(run_dir=run_dir, evaluation_id="taxonomy_metrics_v2")
            combined = json.loads(
                (run_dir / "evaluations" / "taxonomy_metrics_v2" / "evaluation_summary.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(manifest["provider_calls"], 0)
            self.assertEqual(manifest["metric_families"], ["a_box_repair_v1", "tbox_taxonomy_patch_v1"])
            self.assertIsNone(combined["logic_only"]["combined_repair_success_score"])
            self.assertEqual(
                combined["logic_only"]["tbox_taxonomy_patch"]["metric_family"],
                "tbox_taxonomy_patch_v1",
            )
            self.assertEqual(combined["logic_only"]["a_box"]["counts"].get("cases", 0), 0)


if __name__ == "__main__":
    unittest.main()
