import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

from guardian.model_provider import StaticResponseProvider
from guardian.prompts import get_prompt_template
from guardian.reasoning import build_prompt_bundle, build_track_diagnosis_prompt_bundle, run_reasoning_floor


class ReasoningFloorTests(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    def _make_stub_fixture(self) -> tuple[Path, Path, Path, Path, Callable[[dict[str, Any]], dict[str, Any]]]:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        root = Path(tmp_dir.name)
        classified_path = root / "classified.jsonl"
        world_state_path = root / "world_state.json"
        selection_manifest_path = root / "selection.json"

        classified_rows = [
            {
                "id": "repair_case",
                "qid": "Q1",
                "property": "P31",
                "track": "A_BOX",
                "labels_en": {},
                "violation_context": {"value": ["Q2"]},
                "repair_target": {"action": "UPDATE", "old_value": ["Q2"], "new_value": ["Q5"]},
                "persistence_check": {},
                "popularity": {"score": 0.3},
                "classification": {"class": "TypeB", "subtype": "LOCAL_NEIGHBOR_IDS"},
            },
            {
                "id": "reform_case",
                "qid": "Q2",
                "property": "P31",
                "track": "T_BOX",
                "labels_en": {},
                "violation_context": {},
                "repair_target": {
                    "constraint_delta": {
                        "changed_constraint_types": ["Q21510859"],
                        "signature_after": [
                            {
                                "constraint_qid": "Q21510859",
                                "snaktype": "VALUE",
                                "rank": "normal",
                                "qualifiers": [{"property_id": "P2305", "values": ["Q5", "Q43229"]}],
                            }
                        ],
                    }
                },
                "persistence_check": {},
                "popularity": {"score": 0.9},
                "classification": {"class": "T_BOX", "subtype": "RELAXATION_SET_EXPANSION"},
            },
        ]
        self._write_jsonl(classified_path, classified_rows)
        with open(world_state_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "repair_case": {
                        "L1_ego_node": {
                            "qid": "Q1",
                            "label": "Entity",
                            "description": "desc",
                            "properties": {"P31": ["Q2"]},
                        },
                        "L2_labels": {},
                        "L3_neighborhood": {"outgoing_edges": []},
                        "L4_constraints": {"constraints": []},
                    },
                    "reform_case": {
                        "L1_ego_node": {
                            "qid": "Q2",
                            "label": "Entity",
                            "description": "desc",
                            "properties": {},
                        },
                        "L2_labels": {},
                        "L3_neighborhood": {"outgoing_edges": []},
                        "L4_constraints": {"constraints": []},
                        "constraint_change_context": {},
                    },
                },
                fh,
            )
        selection_manifest_path.write_text(
            json.dumps({"selected_case_ids": ["reform_case"]}),
            encoding="utf-8",
        )

        def resolver(metadata: dict[str, Any]) -> dict[str, Any]:
            if metadata["task_type"] == "track_diagnosis":
                return {
                    "case_id": metadata["case_id"],
                    "predicted_track": "T_BOX" if metadata["case_id"] == "reform_case" else "A_BOX",
                    "confidence": "high",
                }
            if metadata["case_id"] == "reform_case":
                return {
                    "case_id": "reform_case",
                    "target": {"pid": "P31", "constraint_type_qid": "Q21510859"},
                    "proposal": {
                        "action": "RELAXATION_SET_EXPANSION",
                        "signature_after": [
                            {
                                "constraint_qid": "Q21510859",
                                "snaktype": "VALUE",
                                "rank": "normal",
                                "qualifiers": [{"property_id": "P2305", "values": ["Q5", "Q43229"]}],
                            }
                        ],
                    },
                }
            return {
                "case_id": "repair_case",
                "target": {"qid": "Q1", "pid": "P31"},
                "ops": [{"op": "SET", "pid": "P31", "value": "Q5"}],
            }

        return root, classified_path, world_state_path, selection_manifest_path, resolver

    def test_reasoning_floor_stub_run(self) -> None:
        root, classified_path, world_state_path, selection_manifest_path, resolver = self._make_stub_fixture()
        summary = run_reasoning_floor(
            classified_path=classified_path,
            world_state_path=world_state_path,
            output_dir=root / "outputs",
            provider=StaticResponseProvider(resolver, model="stub-model"),
            ablation_bundles=["minimal_case"],
            selection_manifest_path=selection_manifest_path,
        )
        self.assertIn("paper_summary", summary)
        self.assertIn("run_info", summary)
        self.assertIn("usage", summary)
        self.assertEqual(summary["counts"]["cases"], 1)
        self.assertEqual(summary["counts"]["track_diagnosis_exact_match"], 1)
        self.assertEqual(summary["run_info"]["model"], "stub-model")
        self.assertEqual(summary["run_info"]["execution_mode"], "sync")
        self.assertEqual(summary["usage"]["prompt_tokens"], 0)
        self.assertEqual(summary["usage"]["completion_tokens"], 0)
        self.assertIn("stub_model", summary["run_info"]["output_dir"])
        self.assertEqual(summary["inputs"]["selection_manifest"], str(selection_manifest_path))

    def test_reasoning_floor_batch_stub_run(self) -> None:
        root, classified_path, world_state_path, selection_manifest_path, resolver = self._make_stub_fixture()
        summary = run_reasoning_floor(
            classified_path=classified_path,
            world_state_path=world_state_path,
            output_dir=root / "outputs",
            provider=StaticResponseProvider(resolver, model="stub-model"),
            ablation_bundles=["minimal_case"],
            selection_manifest_path=selection_manifest_path,
            execution_mode="batch",
            batch_poll_interval_seconds=0.0,
        )
        run_dir = Path(summary["run_info"]["output_dir"])
        manifest_rows = [
            json.loads(line)
            for line in (run_dir / "run_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(summary["counts"]["cases"], 1)
        self.assertEqual(summary["counts"]["track_diagnosis_exact_match"], 1)
        self.assertEqual(summary["run_info"]["execution_mode"], "batch")
        self.assertEqual(summary["run_info"]["batch"]["status"], "completed")
        self.assertTrue((run_dir / "batch_input.jsonl").exists())
        self.assertTrue((run_dir / "batch_request_manifest.jsonl").exists())
        self.assertTrue((run_dir / "static_batch_output.jsonl").exists())
        self.assertTrue(all(row.get("custom_id") for row in manifest_rows))
        self.assertEqual(summary["usage"]["prompt_tokens"], 0)
        self.assertEqual(summary["usage"]["completion_tokens"], 0)

    def test_reasoning_floor_defaults_to_batch_for_openai_provider(self) -> None:
        root, classified_path, world_state_path, selection_manifest_path, resolver = self._make_stub_fixture()
        summary = run_reasoning_floor(
            classified_path=classified_path,
            world_state_path=world_state_path,
            output_dir=root / "outputs",
            provider=StaticResponseProvider(resolver, provider_name="openai", model="stub-model"),
            ablation_bundles=["minimal_case"],
            selection_manifest_path=selection_manifest_path,
            batch_poll_interval_seconds=0.0,
        )
        run_dir = Path(summary["run_info"]["output_dir"])
        self.assertEqual(summary["run_info"]["execution_mode"], "batch")
        self.assertTrue((run_dir / "batch_input.jsonl").exists())
        self.assertTrue((run_dir / "batch_request_manifest.jsonl").exists())

    def test_prompt_bundles_use_named_templates(self) -> None:
        record = {
            "id": "repair_case",
            "qid": "Q1",
            "property": "P31",
            "track": "A_BOX",
            "classification": {"class": "TypeB"},
            "labels_en": {},
            "violation_context": {"value": ["Q2"]},
            "persistence_check": {},
        }
        world_state_entry = {"L4_constraints": {"constraints": []}}

        proposal_bundle = build_prompt_bundle(record, world_state_entry, "logic_only")
        diagnosis_bundle = build_track_diagnosis_prompt_bundle(record, world_state_entry, "logic_only")

        self.assertEqual(proposal_bundle.prompt_name, "reasoning_floor_a_box_zero_shot")
        self.assertEqual(
            proposal_bundle.system_prompt,
            get_prompt_template("reasoning_floor_a_box_zero_shot").system_prompt,
        )
        self.assertEqual(diagnosis_bundle.prompt_name, "reasoning_floor_track_diagnosis_zero_shot")
        self.assertEqual(
            diagnosis_bundle.system_prompt,
            get_prompt_template("reasoning_floor_track_diagnosis_zero_shot").system_prompt,
        )
        self.assertEqual(proposal_bundle.response_format, {"type": "json_object"})
        self.assertIn('"logic_context"', proposal_bundle.prompt)


if __name__ == "__main__":
    unittest.main()
