import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

from guardian.model_provider import StaticResponseProvider
from guardian.prompts import get_prompt_template
from guardian.reasoning import (
    _collect_selected_records_in_order,
    build_prompt_bundle,
    build_track_diagnosis_prompt_bundle,
    run_reasoning_floor,
)


class CostedStaticOpenAIProvider(StaticResponseProvider):
    def generate(
        self,
        prompt: str,
        system_prompt: str,
        response_format: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[Any, Any, dict[str, Any]]:
        raw, parsed, usage = super().generate(prompt, system_prompt, response_format, metadata)
        usage["estimated_cost_usd"] = 1.0
        usage["input_cost_per_1m_tokens_usd"] = 2.0
        usage["output_cost_per_1m_tokens_usd"] = 4.0
        return raw, parsed, usage

    def parse_batch_result(
        self,
        result_record: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[Any, Any, dict[str, Any], str | None]:
        raw, parsed, usage, error = super().parse_batch_result(result_record, metadata)
        usage["estimated_cost_usd"] = 1.0
        usage["input_cost_per_1m_tokens_usd"] = 2.0
        usage["output_cost_per_1m_tokens_usd"] = 4.0
        return raw, parsed, usage, error


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
        self.assertEqual(summary["run_info"]["evaluation"]["classified_record_strategy"], "memory_cache")
        self.assertIsNone(summary["run_info"]["evaluation"]["filtered_classified_path"])
        self.assertEqual(summary["parse_errors"]["proposal_parse_error_count"], 0)

    def test_reasoning_floor_streams_filtered_classified_subset_for_large_eval(self) -> None:
        root, classified_path, world_state_path, _selection_manifest_path, resolver = self._make_stub_fixture()
        with patch("guardian.reasoning.EVALUATION_IN_MEMORY_CASE_THRESHOLD", 1):
            summary = run_reasoning_floor(
                classified_path=classified_path,
                world_state_path=world_state_path,
                output_dir=root / "outputs",
                provider=StaticResponseProvider(resolver, model="stub-model"),
                ablation_bundles=["minimal_case"],
            )

        run_dir = Path(summary["run_info"]["output_dir"])
        filtered_path = run_dir / "selected_classified_records.jsonl"
        evaluation_summary = json.loads((run_dir / "minimal_case" / "evaluation_summary.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["run_info"]["evaluation"]["classified_record_strategy"], "filtered_subset_stream")
        self.assertEqual(summary["run_info"]["evaluation"]["filtered_classified_path"], str(filtered_path))
        self.assertEqual(summary["run_info"]["evaluation"]["filtered_record_count"], 2)
        self.assertTrue(filtered_path.exists())
        filtered_rows = [json.loads(line) for line in filtered_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual([row["id"] for row in filtered_rows], ["repair_case", "reform_case"])
        self.assertEqual(evaluation_summary["inputs"]["classified_benchmark"], str(classified_path))

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
        self.assertEqual(summary["run_info"]["batch"]["mode"], "one_stage")
        self.assertEqual(summary["run_info"]["batch"]["overall_status"], "completed")
        self.assertEqual(summary["run_info"]["batch"]["phases"]["combined"]["status"], "completed")
        self.assertTrue((run_dir / "batch_input.jsonl").exists())
        self.assertTrue((run_dir / "batch_request_manifest.jsonl").exists())
        self.assertTrue((run_dir / "static_batch_output.jsonl").exists())
        self.assertTrue(all(row.get("custom_id") for row in manifest_rows))
        self.assertEqual(summary["usage"]["prompt_tokens"], 0)
        self.assertEqual(summary["usage"]["completion_tokens"], 0)

    def test_reasoning_floor_parallel_stub_run(self) -> None:
        root, classified_path, world_state_path, selection_manifest_path, resolver = self._make_stub_fixture()
        summary = run_reasoning_floor(
            classified_path=classified_path,
            world_state_path=world_state_path,
            output_dir=root / "outputs",
            provider=StaticResponseProvider(resolver, model="stub-model"),
            ablation_bundles=["minimal_case"],
            selection_manifest_path=selection_manifest_path,
            execution_mode="parallel",
            parallel_workers=2,
        )

        self.assertEqual(summary["counts"]["cases"], 1)
        self.assertEqual(summary["counts"]["track_diagnosis_exact_match"], 1)
        self.assertEqual(summary["run_info"]["execution_mode"], "parallel")
        self.assertEqual(summary["run_info"]["parallel"]["workers"], 1)
        self.assertEqual(summary["run_info"]["parallel"]["source"], "argument")
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

    def test_reasoning_floor_batch_applies_openai_cost_discount(self) -> None:
        root, classified_path, world_state_path, selection_manifest_path, resolver = self._make_stub_fixture()
        summary = run_reasoning_floor(
            classified_path=classified_path,
            world_state_path=world_state_path,
            output_dir=root / "outputs",
            provider=CostedStaticOpenAIProvider(resolver, provider_name="openai", model="stub-model"),
            ablation_bundles=["minimal_case"],
            selection_manifest_path=selection_manifest_path,
            execution_mode="batch",
            batch_poll_interval_seconds=0.0,
        )
        self.assertTrue(summary["run_info"]["batch_mode_used"])
        self.assertTrue(summary["usage"]["batch_pricing_applied"])
        self.assertEqual(summary["usage"]["cost_estimation_mode"], "openai_batch_discount_applied")
        self.assertEqual(summary["usage"]["cost_estimation_multiplier"], 0.5)
        self.assertEqual(summary["usage"]["estimated_cost_usd"], 1.0)

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
        self.assertNotIn('"classification"', proposal_bundle.prompt)
        self.assertNotIn('"persistence_check"', proposal_bundle.prompt)
        self.assertNotIn('"repair_target"', proposal_bundle.prompt)
        self.assertNotIn('"track":', proposal_bundle.prompt)

    def test_t_box_prompt_bundle_prunes_local_graph_context(self) -> None:
        record = {
            "id": "reform_case",
            "qid": "Q2",
            "property": "P31",
            "track": "T_BOX",
            "classification": {"class": "T_BOX"},
            "labels_en": {},
            "violation_context": {
                "report_violation_type": "one-of",
                "report_violation_type_normalized": "one-of",
                "value": ["Q5"],
            },
            "persistence_check": {"truth_tokens": ["leak"]},
        }
        world_state_entry = {
            "L1_ego_node": {
                "qid": "Q2",
                "label": "Entity",
                "description": "desc",
                "properties": {"P31": ["Q5"], "P279": ["Q35120"]},
            },
            "L2_labels": {
                "entities": {
                    "Q2": {"label": "Entity"},
                    "Q5": {"label": "human"},
                    "Q35120": {"label": "entity"},
                    "Q999": {"label": "unrelated"},
                }
            },
            "L3_neighborhood": {
                "outgoing_edges": [
                    {"pid": "P279", "target": "Q35120"},
                    {"pid": "P31", "target": "Q5"},
                    {"pid": "P999", "target": "Q999"},
                ]
            },
            "L4_constraints": {
                "constraints": [
                    {
                        "constraint_type": {"qid": "Q21510859"},
                        "qualifiers": [{"property_id": "P2305", "values": ["Q5"]}],
                    },
                    {
                        "constraint_type": {"qid": "Q21503250"},
                        "qualifiers": [{"property_id": "P2305", "values": ["Q999"]}],
                    },
                ]
            },
        }

        proposal_bundle = build_prompt_bundle(record, world_state_entry, "local_graph")
        diagnosis_bundle = build_track_diagnosis_prompt_bundle(record, world_state_entry, "local_graph")

        self.assertIn('"local_context"', proposal_bundle.prompt)
        self.assertIn('"L4_constraints"', proposal_bundle.prompt)
        self.assertIn('"L2_labels"', proposal_bundle.prompt)
        self.assertIn('"L3_neighborhood"', proposal_bundle.prompt)
        self.assertIn('"P31"', proposal_bundle.prompt)
        self.assertNotIn('"P999"', proposal_bundle.prompt)
        self.assertNotIn('"Q999"', proposal_bundle.prompt)
        self.assertNotIn('"persistence_check"', proposal_bundle.prompt)
        self.assertIn('"L2_labels"', diagnosis_bundle.prompt)

    def test_t_box_prompt_template_avoids_specific_anchor_example(self) -> None:
        template = get_prompt_template("reasoning_floor_t_box_zero_shot")
        self.assertNotIn("Q21510859", template.user_prompt_template)
        self.assertNotIn("Q43229", template.user_prompt_template)
        self.assertIn("constraint-family QIDs from the supplied constraint context", template.user_prompt_template)
        self.assertIn("Do not copy violating entity or type QIDs into constraint_type_qid", template.user_prompt_template)

    def test_track_diagnosis_prompt_template_includes_schema_vs_claim_guidance(self) -> None:
        template = get_prompt_template("reasoning_floor_track_diagnosis_zero_shot")
        self.assertIn("Allowed-entity-types, property-scope, one-of, range", template.user_prompt_template)
        self.assertIn("If a property currently allows only certain entity types", template.user_prompt_template)
        self.assertIn("predict AMBIGUOUS", template.user_prompt_template)

    def test_reasoning_floor_preserves_manifest_order_before_max_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            classified_path = root / "classified.jsonl"
            world_state_path = root / "world_state.json"
            selection_manifest_path = root / "selection.json"

            classified_rows = [
                {
                    "id": "case_a",
                    "qid": "Q1",
                    "property": "P31",
                    "track": "A_BOX",
                    "labels_en": {},
                    "violation_context": {"value": ["Q2"]},
                    "repair_target": {"action": "UPDATE", "old_value": ["Q2"], "new_value": ["Q3"]},
                    "classification": {"class": "TypeA", "subtype": "DIRECT_VALUE"},
                },
                {
                    "id": "case_b",
                    "qid": "Q2",
                    "property": "P31",
                    "track": "A_BOX",
                    "labels_en": {},
                    "violation_context": {"value": ["Q4"]},
                    "repair_target": {"action": "UPDATE", "old_value": ["Q4"], "new_value": ["Q5"]},
                    "classification": {"class": "TypeA", "subtype": "DIRECT_VALUE"},
                },
                {
                    "id": "case_c",
                    "qid": "Q3",
                    "property": "P31",
                    "track": "A_BOX",
                    "labels_en": {},
                    "violation_context": {"value": ["Q6"]},
                    "repair_target": {"action": "UPDATE", "old_value": ["Q6"], "new_value": ["Q7"]},
                    "classification": {"class": "TypeA", "subtype": "DIRECT_VALUE"},
                },
            ]
            self._write_jsonl(classified_path, classified_rows)
            world_state_path.write_text(
                json.dumps(
                    {
                        row["id"]: {
                            "L1_ego_node": {"qid": row["qid"], "properties": {"P31": row["violation_context"]["value"]}},
                            "L4_constraints": {"constraints": []},
                        }
                        for row in classified_rows
                    }
                ),
                encoding="utf-8",
            )
            selection_manifest_path.write_text(
                json.dumps({"selected_case_ids": ["case_c", "case_a", "case_b"]}),
                encoding="utf-8",
            )

            def resolver(metadata: dict[str, Any]) -> dict[str, Any]:
                if metadata["task_type"] == "track_diagnosis":
                    return {"case_id": metadata["case_id"], "predicted_track": "A_BOX", "confidence": "high"}
                return {
                    "case_id": metadata["case_id"],
                    "target": {"qid": {"case_a": "Q1", "case_b": "Q2", "case_c": "Q3"}[metadata["case_id"]], "pid": "P31"},
                    "ops": [{"op": "SET", "pid": "P31", "value": {"case_a": "Q3", "case_b": "Q5", "case_c": "Q7"}[metadata["case_id"]]}],
                }

            summary = run_reasoning_floor(
                classified_path=classified_path,
                world_state_path=world_state_path,
                output_dir=root / "outputs",
                provider=StaticResponseProvider(resolver, model="stub-model"),
                ablation_bundles=["minimal_case"],
                selection_manifest_path=selection_manifest_path,
                max_cases=2,
            )

            run_dir = Path(summary["run_info"]["output_dir"])
            manifest_rows = [
                json.loads(line)
                for line in (run_dir / "run_manifest.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            proposal_case_ids = [row["case_id"] for row in manifest_rows if row["task_type"] == "proposal"]
            self.assertEqual(proposal_case_ids, ["case_c", "case_a"])

    def test_collect_selected_records_limits_manifest_scan_before_reading_tail_records(self) -> None:
        rows = [
            {"id": "case_a", "track": "A_BOX"},
            {"id": "case_c", "track": "A_BOX"},
            {"id": "tail_1", "track": "A_BOX"},
            {"id": "tail_2", "track": "A_BOX"},
            {"id": "case_b", "track": "A_BOX"},
        ]
        yielded_ids: list[str] = []

        def fake_iter_jsonl(_path: str | Path) -> Any:
            for row in rows:
                yielded_ids.append(row["id"])
                yield row

        with patch("guardian.reasoning.iter_jsonl", side_effect=fake_iter_jsonl):
            selected = _collect_selected_records_in_order(
                "ignored.jsonl",
                case_ids=["case_c", "case_a", "case_b"],
                max_cases=2,
            )

        self.assertEqual([row["id"] for row in selected], ["case_c", "case_a"])
        self.assertEqual(yielded_ids, ["case_a", "case_c"])

    def test_reasoning_floor_diagnosis_routed_skips_ambiguous_proposals(self) -> None:
        root, classified_path, world_state_path, selection_manifest_path, _resolver = self._make_stub_fixture()

        def resolver(metadata: dict[str, Any]) -> dict[str, Any]:
            if metadata["task_type"] == "track_diagnosis":
                return {
                    "case_id": metadata["case_id"],
                    "predicted_track": "AMBIGUOUS" if metadata["case_id"] == "reform_case" else "A_BOX",
                    "confidence": "medium",
                }
            return {
                "case_id": metadata["case_id"],
                "target": {"qid": "Q1", "pid": "P31"},
                "ops": [{"op": "SET", "pid": "P31", "value": "Q5"}],
            }

        summary = run_reasoning_floor(
            classified_path=classified_path,
            world_state_path=world_state_path,
            output_dir=root / "outputs",
            provider=StaticResponseProvider(resolver, model="stub-model"),
            ablation_bundles=["minimal_case"],
            selection_manifest_path=selection_manifest_path,
            proposal_track_mode="diagnosis_routed",
        )

        run_dir = Path(summary["run_info"]["output_dir"])
        manifest_rows = [
            json.loads(line)
            for line in (run_dir / "run_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        proposal_row = next(row for row in manifest_rows if row["case_id"] == "reform_case" and row["task_type"] == "proposal")
        self.assertEqual(summary["run_info"]["proposal_track_mode"], "diagnosis_routed")
        self.assertEqual(proposal_row["parse_status"], "skipped_ambiguous_track")
        self.assertEqual(proposal_row["proposal_track_used"], "AMBIGUOUS")
        t_box_rows = [
            json.loads(line)
            for line in (run_dir / "minimal_case" / "t_box_proposals.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(t_box_rows, [])

    def test_reasoning_floor_diagnosis_routed_parallel_stub_run(self) -> None:
        root, classified_path, world_state_path, selection_manifest_path, resolver = self._make_stub_fixture()
        summary = run_reasoning_floor(
            classified_path=classified_path,
            world_state_path=world_state_path,
            output_dir=root / "outputs",
            provider=StaticResponseProvider(resolver, model="stub-model"),
            ablation_bundles=["minimal_case"],
            selection_manifest_path=selection_manifest_path,
            execution_mode="parallel",
            parallel_workers=2,
            proposal_track_mode="diagnosis_routed",
        )

        self.assertEqual(summary["counts"]["cases"], 1)
        self.assertEqual(summary["run_info"]["execution_mode"], "parallel")
        self.assertEqual(summary["run_info"]["proposal_track_mode"], "diagnosis_routed")
        self.assertEqual(summary["counts"]["track_diagnosis_exact_match"], 1)

    def test_reasoning_floor_diagnosis_routed_batch_uses_two_stage_artifacts(self) -> None:
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
            proposal_track_mode="diagnosis_routed",
        )

        run_dir = Path(summary["run_info"]["output_dir"])
        self.assertEqual(summary["run_info"]["batch"]["mode"], "two_stage")
        self.assertEqual(summary["run_info"]["batch"]["overall_status"], "completed")
        self.assertIn("diagnosis", summary["run_info"]["batch"]["phases"])
        self.assertIn("proposal", summary["run_info"]["batch"]["phases"])
        self.assertTrue((run_dir / "diagnosis_batch_input.jsonl").exists())
        self.assertTrue((run_dir / "proposal_batch_input.jsonl").exists())
        self.assertTrue((run_dir / "diagnosis_static_batch_output.jsonl").exists())
        self.assertTrue((run_dir / "proposal_static_batch_output.jsonl").exists())

    def test_reasoning_floor_normalizes_non_list_provenance_outputs(self) -> None:
        root, classified_path, world_state_path, _selection_manifest_path, _resolver = self._make_stub_fixture()

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
                    "provenance": {"node_id": "Q21510859", "snippet": "constraint"},
                }
            return {
                "case_id": "repair_case",
                "target": {"qid": "Q1", "pid": "P31"},
                "ops": [{"op": "SET", "pid": "P31", "value": "Q5"}],
                "provenance": "historical statement",
            }

        summary = run_reasoning_floor(
            classified_path=classified_path,
            world_state_path=world_state_path,
            output_dir=root / "outputs",
            provider=StaticResponseProvider(resolver, model="stub-model"),
            ablation_bundles=["minimal_case"],
        )

        run_dir = Path(summary["run_info"]["output_dir"])
        a_box_rows = [
            json.loads(line)
            for line in (run_dir / "minimal_case" / "a_box_proposals.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        t_box_rows = [
            json.loads(line)
            for line in (run_dir / "minimal_case" / "t_box_proposals.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(summary["parse_errors"]["proposal_parse_error_count"], 0)
        self.assertEqual(a_box_rows[0]["provenance"], [{"kind": "OTHER", "snippet": "historical statement"}])
        self.assertEqual(t_box_rows[0]["provenance"], [{"kind": "KG", "node_id": "Q21510859", "snippet": "constraint"}])

    def test_reasoning_floor_rejects_invalid_t_box_constraint_family_qids(self) -> None:
        root, classified_path, world_state_path, selection_manifest_path, _resolver = self._make_stub_fixture()

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
                    "target": {"pid": "P31", "constraint_type_qid": "Q11122"},
                    "proposal": {
                        "action": "RELAXATION_SET_EXPANSION",
                        "signature_after": [
                            {
                                "constraint_qid": "Q11122",
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

        summary = run_reasoning_floor(
            classified_path=classified_path,
            world_state_path=world_state_path,
            output_dir=root / "outputs",
            provider=StaticResponseProvider(resolver, model="stub-model"),
            ablation_bundles=["minimal_case"],
            selection_manifest_path=selection_manifest_path,
        )

        run_dir = Path(summary["run_info"]["output_dir"])
        manifest_rows = [
            json.loads(line)
            for line in (run_dir / "run_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        proposal_row = next(row for row in manifest_rows if row["case_id"] == "reform_case" and row["task_type"] == "proposal")
        t_box_rows = [
            json.loads(line)
            for line in (run_dir / "minimal_case" / "t_box_proposals.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertEqual(summary["parse_errors"]["proposal_parse_error_count"], 1)
        self.assertEqual(proposal_row["parse_status"], "parse_error")
        self.assertEqual(proposal_row["parser_error"], "invalid constraint_type_qid for T-box proposal")
        self.assertEqual(t_box_rows, [])


if __name__ == "__main__":
    unittest.main()
