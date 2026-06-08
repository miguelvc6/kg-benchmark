from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from guardian.model_provider import StaticResponseProvider
from lib.prompt_dev import (
    PromptDevEvaluateOptions,
    PromptDevMatrixOptions,
    PromptDevRenderOptions,
    build_prompt_dev_matrix,
    evaluate_prompt_dev_prompts,
    render_prompt_dev_prompts,
    select_examples,
)
from scripts.prompt_dev_templates import render_prompt_dev_prompt


def _abox_record(case_id: str, qid: str, pid: str = "P1", subtype: str = "LOCAL_TEXT_CONFIRMED") -> dict:
    return {
        "id": case_id,
        "track": "A_BOX",
        "qid": qid,
        "property": pid,
        "labels_en": {"qid": "Example item", "property": "Example property"},
        "violation_context": {"value": "Q1", "report_violation_type_qids": ["Q21503250"]},
        "repair_target": {"kind": "A_BOX", "action": "UPDATE", "old_value": ["Q1"], "new_value": ["Q5"]},
        "classification": {
            "class": "TypeB",
            "subtype": subtype,
            "confidence": "high",
            "constraint_types": [{"qid": "Q21503250"}],
            "diagnostics": {"truth_source": "repair_target.new_value", "truth_tokens": ["Q5"]},
        },
        "popularity": {"bucket": "mid", "score": 0.5},
    }


def _tbox_record(case_id: str, qid: str, pid: str = "P2", revision: str = "r1") -> dict:
    signature = [
        {
            "constraint_qid": "Q21510859",
            "snaktype": "VALUE",
            "rank": "normal",
            "qualifiers": [{"property_id": "P2305", "values": ["Q5"]}],
        }
    ]
    return {
        "id": case_id,
        "track": "T_BOX",
        "qid": qid,
        "property": pid,
        "labels_en": {"qid": "Example item", "property": "Example property"},
        "violation_context": {"value": "Q1", "report_violation_type_qids": ["Q21510859"]},
        "repair_target": {
            "kind": "T_BOX",
            "property_revision_id": revision,
            "constraint_delta": {
                "changed_constraint_types": ["Q21510859"],
                "signature_after": signature,
            },
        },
        "classification": {
            "class": "T_BOX",
            "subtype": "RELAXATION_SET_EXPANSION",
            "confidence": "high",
            "constraint_types": [{"qid": "Q21510859"}],
            "diagnostics": {"truth_source": "constraint_delta", "truth_tokens": ["Q5"]},
        },
        "popularity": {"bucket": "mid", "score": 0.5},
    }


def _manifest(case_ids: list[str], records: list[dict]) -> dict:
    annotations = {}
    for record in records:
        classification = record.get("classification", {})
        subtype = classification.get("subtype") or "unknown"
        prefix = "DEV_TBOX" if record.get("track") == "T_BOX" else "DEV_ABOX"
        annotations[record["id"]] = {
            "selection_stratum": f"{prefix}_{subtype}",
            "group_key": f"group:{record['id']}",
            "tbox_revision_key": record.get("repair_target", {}).get("property_revision_id"),
        }
    return {"selected_case_ids": case_ids, "case_annotations": annotations}


class PromptDevTests(unittest.TestCase):
    def test_matrix_expands_axes_without_inference_fields(self) -> None:
        matrix = build_prompt_dev_matrix(
            PromptDevMatrixOptions(
                representations=("hybrid_json_nl", "pure_nl"),
                example_policies=("zero_shot",),
                context_bundles=("logic_only",),
                tasks=("track_diagnosis", "repair_proposal"),
                repair_track_modes=("oracle", "diagnosis_routed"),
            )
        )

        self.assertEqual(matrix["counts"]["rows"], 6)
        self.assertTrue(all(row["run_scope"] == "dev_only" for row in matrix["rows"]))
        self.assertIn("parse_validity", matrix["rows"][0]["metrics"])

    def test_few_shot_selection_excludes_same_case_qid_property_and_core(self) -> None:
        eval_record = _abox_record("eval", "Q1", "P1")
        same_qid = _abox_record("same_qid", "Q1", "P9")
        same_property = _abox_record("same_property", "Q2", "P1")
        blocked_core = _abox_record("blocked_core", "Q3", "P3")
        usable = _abox_record("usable", "Q4", "P4")

        examples = select_examples(
            eval_record=eval_record,
            candidate_records=[same_qid, same_property, blocked_core, usable],
            policy="matched_2shot",
            task="a_box_repair",
            seed=13,
            blocked_core={"case_ids": {"blocked_core"}, "group_keys": set(), "tbox_revision_keys": set()},
        )

        self.assertEqual([example["case_id"] for example in examples], ["usable"])
        self.assertEqual(examples[0]["output_payload"]["target"]["qid"], "Q4")

    def test_template_rendering_keeps_contract_and_examples_visible(self) -> None:
        prompt = render_prompt_dev_prompt(
            task="track_diagnosis",
            representation="compact_table",
            case_payload={"id": "case_1", "qid": "Q1", "property": "P1", "violation_context": {"value": "Q2"}},
            examples=[
                {
                    "input_payload": {
                        "id": "case_0",
                        "qid": "Q0",
                        "property": "P1",
                        "violation_context": {"value": "Q2"},
                    },
                    "output_payload": {"case_id": "case_0", "predicted_track": "A_BOX"},
                }
            ],
        )

        self.assertIn("predicted_track", prompt.user_prompt)
        self.assertIn("Example 1 input", prompt.user_prompt)
        self.assertIn("case.id", prompt.user_prompt)

    def test_render_prompts_writes_prompt_artifacts_without_provider_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            classified = root / "classified.jsonl"
            records = [_abox_record("eval", "Q1", "P1"), _tbox_record("example", "Q2", "P2")]
            classified.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
            manifest = root / "dev.json"
            manifest.write_text(json.dumps(_manifest(["eval", "example"], records)), encoding="utf-8")
            world_state = root / "world.json"
            world_state.write_text("{}", encoding="utf-8")

            summary = render_prompt_dev_prompts(
                PromptDevRenderOptions(
                    classified_benchmark=classified,
                    world_state=world_state,
                    dev_manifest=manifest,
                    output_dir=root / "out",
                    max_cases=1,
                    representations=("hybrid_json_nl",),
                    example_policies=("zero_shot",),
                    context_bundles=("minimal_case",),
                    tasks=("track_diagnosis", "repair_proposal"),
                    repair_track_modes=("oracle",),
                )
            )

            self.assertEqual(summary["counts"]["rendered_prompts"], 2)
            self.assertEqual(summary["counts"]["by_track"], {"A_BOX": 1})
            self.assertTrue((root / "out" / "prompt_dev_rendered_prompts.jsonl").exists())
            self.assertTrue((root / "out" / "prompt_dev_prompt_review.md").exists())

    def test_stratified_max_cases_includes_a_box_and_t_box_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            tbox_records = [_tbox_record(f"reform_tbox_{index:03d}", f"Q{index}", "P2") for index in range(30)]
            abox_records = [_abox_record(f"repair_abox_{index:03d}", f"Q{index + 100}", "P1") for index in range(5)]
            records = tbox_records + abox_records
            classified = root / "classified.jsonl"
            classified.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
            manifest = root / "dev.json"
            manifest.write_text(json.dumps(_manifest([record["id"] for record in records], records)), encoding="utf-8")
            world_state = root / "world.json"
            world_state.write_text("{}", encoding="utf-8")

            summary = render_prompt_dev_prompts(
                PromptDevRenderOptions(
                    classified_benchmark=classified,
                    world_state=world_state,
                    dev_manifest=manifest,
                    output_dir=root / "out",
                    max_cases=24,
                    representations=("hybrid_json_nl",),
                    example_policies=("zero_shot",),
                    context_bundles=("minimal_case",),
                    tasks=("track_diagnosis",),
                )
            )

            self.assertGreater(summary["counts"]["by_track"].get("A_BOX", 0), 0)
            self.assertGreater(summary["counts"]["by_track"].get("T_BOX", 0), 0)

    def test_rendered_prompt_text_uses_neutral_case_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            records = [_abox_record("repair_case_raw", "Q1", "P1"), _tbox_record("reform_case_raw", "Q2", "P2")]
            classified = root / "classified.jsonl"
            classified.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
            manifest = root / "dev.json"
            manifest.write_text(json.dumps(_manifest([record["id"] for record in records], records)), encoding="utf-8")
            world_state = root / "world.json"
            world_state.write_text("{}", encoding="utf-8")

            render_prompt_dev_prompts(
                PromptDevRenderOptions(
                    classified_benchmark=classified,
                    world_state=world_state,
                    dev_manifest=manifest,
                    output_dir=root / "out",
                    max_cases=2,
                    representations=("hybrid_json_nl",),
                    example_policies=("zero_shot",),
                    context_bundles=("minimal_case",),
                    tasks=("track_diagnosis",),
                )
            )

            rendered_rows = [
                json.loads(line)
                for line in (root / "out" / "prompt_dev_rendered_prompts.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertTrue(all(row["visible_case_id"].startswith("case_") for row in rendered_rows))
            prompt_text = "\n".join(row["system_prompt"] + "\n" + row["user_prompt"] for row in rendered_rows)
            self.assertNotIn("repair_case_raw", prompt_text)
            self.assertNotIn("reform_case_raw", prompt_text)

    def test_local_graph_prompt_payload_omits_sitelinks_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            record = _abox_record("repair_case_raw", "Q1", "P1")
            classified = root / "classified.jsonl"
            classified.write_text(json.dumps(record) + "\n", encoding="utf-8")
            manifest = root / "dev.json"
            manifest.write_text(json.dumps(_manifest([record["id"]], [record])), encoding="utf-8")
            world_state = root / "world.json"
            world_state.write_text(
                json.dumps(
                    {
                        "repair_case_raw": {
                            "L1_ego_node": {
                                "qid": "Q1",
                                "label": "Example",
                                "description": "Example item",
                                "sitelinks_count": 999,
                                "properties": {"P1": [{"value": "Q1"}]},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            render_prompt_dev_prompts(
                PromptDevRenderOptions(
                    classified_benchmark=classified,
                    world_state=world_state,
                    dev_manifest=manifest,
                    output_dir=root / "out",
                    max_cases=1,
                    representations=("hybrid_json_nl",),
                    example_policies=("zero_shot",),
                    context_bundles=("local_graph",),
                    tasks=("track_diagnosis",),
                )
            )

            prompt_text = (root / "out" / "prompt_dev_rendered_prompts.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("sitelinks_count", prompt_text)

    def test_tbox_prompts_use_pre_reform_constraints_not_current_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            record = _tbox_record("reform_case_raw", "Q2", "P2")
            record["repair_target"]["constraint_delta"]["signature_before"] = [
                {
                    "constraint_qid": "Q21510859",
                    "snaktype": "VALUE",
                    "rank": "normal",
                    "qualifiers": [{"property_id": "P2305", "values": ["Q_BEFORE"]}],
                }
            ]
            classified = root / "classified.jsonl"
            classified.write_text(json.dumps(record) + "\n", encoding="utf-8")
            manifest = root / "dev.json"
            manifest.write_text(json.dumps(_manifest([record["id"]], [record])), encoding="utf-8")
            world_state = root / "world.json"
            world_state.write_text(
                json.dumps(
                    {
                        "reform_case_raw": {
                            "L4_constraints": {
                                "constraints": [
                                    {
                                        "constraint_type": {"qid": "Q21510859", "label": "allowed entity types"},
                                        "qualifiers": [{"property_id": "P2305", "values": ["Q_AFTER"]}],
                                    }
                                ]
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            render_prompt_dev_prompts(
                PromptDevRenderOptions(
                    classified_benchmark=classified,
                    world_state=world_state,
                    dev_manifest=manifest,
                    output_dir=root / "out",
                    max_cases=1,
                    representations=("hybrid_json_nl",),
                    example_policies=("zero_shot",),
                    context_bundles=("logic_only",),
                    tasks=("repair_proposal",),
                    repair_track_modes=("oracle",),
                )
            )

            prompt_text = (root / "out" / "prompt_dev_rendered_prompts.jsonl").read_text(encoding="utf-8")
            self.assertIn("Q_BEFORE", prompt_text)
            self.assertNotIn("Q_AFTER", prompt_text)

    def test_few_shot_requires_core_manifest_unless_explicitly_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            records = [_abox_record("eval", "Q1", "P1"), _abox_record("example", "Q2", "P2")]
            classified = root / "classified.jsonl"
            classified.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
            manifest = root / "dev.json"
            manifest.write_text(json.dumps(_manifest([record["id"] for record in records], records)), encoding="utf-8")
            world_state = root / "world.json"
            world_state.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "requires --core-manifest"):
                render_prompt_dev_prompts(
                    PromptDevRenderOptions(
                        classified_benchmark=classified,
                        world_state=world_state,
                        dev_manifest=manifest,
                        output_dir=root / "out",
                        max_cases=1,
                        representations=("hybrid_json_nl",),
                        example_policies=("matched_2shot",),
                        context_bundles=("minimal_case",),
                        tasks=("track_diagnosis",),
                    )
                )

    def test_evaluate_prompts_writes_dev_scoring_artifacts_with_static_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            classified = root / "classified.jsonl"
            records = [_abox_record("abox", "Q1", "P1"), _tbox_record("tbox", "Q2", "P2")]
            classified.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
            manifest = root / "dev.json"
            manifest.write_text(json.dumps({"selected_case_ids": ["abox", "tbox"]}), encoding="utf-8")
            world_state = root / "world.json"
            world_state.write_text("{}", encoding="utf-8")

            def resolver(metadata: dict) -> dict:
                if metadata["task_type"] == "track_diagnosis":
                    return {
                        "case_id": metadata["visible_case_id"],
                        "predicted_track": metadata["historical_track"],
                        "confidence": "high",
                        "rationale": "static test diagnosis",
                    }
                if metadata["proposal_track_used"] == "T_BOX":
                    return {
                        "case_id": metadata["visible_case_id"],
                        "target": {"pid": "P2", "constraint_type_qid": "Q21510859"},
                        "proposal": {
                            "action": "RELAXATION_SET_EXPANSION",
                            "signature_after": [
                                {
                                    "constraint_qid": "Q21510859",
                                    "snaktype": "VALUE",
                                    "rank": "normal",
                                    "qualifiers": [{"property_id": "P2305", "values": ["Q5"]}],
                                }
                            ],
                        },
                    }
                return {
                    "case_id": metadata["visible_case_id"],
                    "target": {"qid": "Q1", "pid": "P1"},
                    "ops": [{"op": "SET", "pid": "P1", "value": "Q5", "rank": "normal"}],
                }

            progress_events = []
            summary = evaluate_prompt_dev_prompts(
                PromptDevEvaluateOptions(
                    classified_benchmark=classified,
                    world_state=world_state,
                    dev_manifest=manifest,
                    output_dir=root / "eval",
                    max_cases=2,
                    representations=("hybrid_json_nl",),
                    example_policies=("zero_shot",),
                    context_bundles=("minimal_case",),
                    tasks=("track_diagnosis", "repair_proposal"),
                    repair_track_modes=("oracle",),
                    progress_callback=progress_events.append,
                ),
                provider=StaticResponseProvider(resolver, provider_name="static", model="static-model"),
            )

            self.assertEqual(summary["counts"]["evaluated_prompts"], 4)
            self.assertEqual(progress_events[0]["event"], "start")
            self.assertEqual(progress_events[0]["total"], 4)
            self.assertEqual(Counter(event["event"] for event in progress_events)["advance"], 4)
            self.assertEqual(summary["counts"]["matrix_rows"], 2)
            self.assertTrue((root / "eval" / "prompt_dev_evaluation_summary.json").exists())
            self.assertTrue((root / "eval" / "prompt_dev_evaluation_comparison.md").exists())
            for result in summary["results"]:
                matrix_dir = Path(result["output_dir"])
                self.assertTrue((matrix_dir / "run_manifest.jsonl").exists())
                self.assertTrue((matrix_dir / "evaluation_summary.json").exists())
                if result["task"] == "repair_proposal":
                    self.assertEqual(result["request_errors"]["track_diagnosis_request_error_count"], 0)

    def test_evaluate_prompts_can_retry_existing_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            classified = root / "classified.jsonl"
            records = [_tbox_record("tbox", "Q2", "P2")]
            classified.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
            manifest = root / "dev.json"
            manifest.write_text(json.dumps({"selected_case_ids": ["tbox"]}), encoding="utf-8")
            world_state = root / "world.json"
            world_state.write_text("{}", encoding="utf-8")
            options = PromptDevEvaluateOptions(
                classified_benchmark=classified,
                world_state=world_state,
                dev_manifest=manifest,
                output_dir=root / "eval",
                max_cases=1,
                representations=("hybrid_json_nl",),
                example_policies=("zero_shot",),
                context_bundles=("minimal_case",),
                tasks=("track_diagnosis",),
            )

            evaluate_prompt_dev_prompts(
                options,
                provider=StaticResponseProvider(lambda _metadata: "not-json", provider_name="static", model="static"),
            )
            skipped_summary = evaluate_prompt_dev_prompts(
                options,
                provider=StaticResponseProvider(
                    lambda metadata: {
                        "case_id": metadata["case_id"],
                        "predicted_track": metadata["historical_track"],
                        "confidence": "high",
                    },
                    provider_name="static",
                    model="static",
                ),
            )
            retried_summary = evaluate_prompt_dev_prompts(
                PromptDevEvaluateOptions(
                    **{**options.__dict__, "retry_failures": True}
                ),
                provider=StaticResponseProvider(
                    lambda metadata: {
                        "case_id": metadata["case_id"],
                        "predicted_track": metadata["historical_track"],
                        "confidence": "high",
                    },
                    provider_name="static",
                    model="static",
                ),
            )

            self.assertEqual(skipped_summary["counts"]["by_parse_status"], {"skipped_existing_parse_error": 1})
            self.assertEqual(retried_summary["counts"]["by_parse_status"], {"normalized": 1})
            matrix_dir = Path(retried_summary["results"][0]["output_dir"])
            self.assertTrue((matrix_dir / "track_diagnoses.jsonl").exists())

    def test_evaluate_prompts_can_skip_oversized_prompts_before_provider_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            classified = root / "classified.jsonl"
            classified.write_text(json.dumps(_tbox_record("tbox", "Q2", "P2")) + "\n", encoding="utf-8")
            manifest = root / "dev.json"
            manifest.write_text(json.dumps({"selected_case_ids": ["tbox"]}), encoding="utf-8")
            world_state = root / "world.json"
            world_state.write_text("{}", encoding="utf-8")

            def fail_if_called(_metadata: dict) -> dict:
                raise AssertionError("provider should not be called for oversized prompts")

            summary = evaluate_prompt_dev_prompts(
                PromptDevEvaluateOptions(
                    classified_benchmark=classified,
                    world_state=world_state,
                    dev_manifest=manifest,
                    output_dir=root / "eval",
                    max_cases=1,
                    representations=("hybrid_json_nl",),
                    example_policies=("zero_shot",),
                    context_bundles=("minimal_case",),
                    tasks=("track_diagnosis",),
                    max_prompt_chars=1,
                ),
                provider=StaticResponseProvider(fail_if_called, provider_name="static", model="static"),
            )

            self.assertEqual(summary["counts"]["by_parse_status"], {"request_error": 1})
            matrix_dir = Path(summary["results"][0]["output_dir"])
            manifest_row = json.loads((matrix_dir / "run_manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("exceeds --max-prompt-chars", manifest_row["provider_error"])

    def test_evaluate_writes_top_level_summary_when_matrix_evaluation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            record = _abox_record("repair_case_raw", "Q1", "P1")
            classified = root / "classified.jsonl"
            classified.write_text(json.dumps(record) + "\n", encoding="utf-8")
            manifest = root / "dev.json"
            manifest.write_text(json.dumps(_manifest([record["id"]], [record])), encoding="utf-8")
            world_state = root / "world.json"
            world_state.write_text("{}", encoding="utf-8")

            with patch("lib.prompt_dev.evaluate_benchmark", side_effect=RuntimeError("scoring failed")):
                summary = evaluate_prompt_dev_prompts(
                    PromptDevEvaluateOptions(
                        classified_benchmark=classified,
                        world_state=world_state,
                        dev_manifest=manifest,
                        output_dir=root / "eval",
                        max_cases=1,
                        representations=("hybrid_json_nl",),
                        example_policies=("zero_shot",),
                        context_bundles=("minimal_case",),
                        tasks=("track_diagnosis",),
                    ),
                    provider=StaticResponseProvider(
                        lambda metadata: {
                            "case_id": metadata["visible_case_id"],
                            "predicted_track": metadata["historical_track"],
                            "confidence": "high",
                        },
                        provider_name="static",
                        model="static",
                    ),
                )

            self.assertTrue((root / "eval" / "prompt_dev_evaluation_summary.json").exists())
            self.assertTrue((root / "eval" / "prompt_dev_evaluation_comparison.md").exists())
            self.assertIn("evaluation_error", summary["results"][0])
            self.assertEqual(summary["results"][0]["counts"]["normalized"], 1)

    def test_diverse_stratified_prefers_qid_and_property_diversity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repeated = [
                _abox_record(f"repair_repeat_{index:03d}", "Q_REPEAT", "P_REPEAT")
                for index in range(8)
            ]
            diverse = [
                _abox_record(f"repair_diverse_{index:03d}", f"Q_DIVERSE_{index}", f"P_DIVERSE_{index}")
                for index in range(8)
            ]
            tbox_records = [
                _tbox_record(f"reform_tbox_{index:03d}", f"Q_TBOX_{index}", f"P_TBOX_{index}")
                for index in range(8)
            ]
            records = repeated + diverse + tbox_records
            classified = root / "classified.jsonl"
            classified.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
            manifest = root / "dev.json"
            manifest.write_text(json.dumps(_manifest([record["id"] for record in records], records)), encoding="utf-8")
            world_state = root / "world.json"
            world_state.write_text("{}", encoding="utf-8")

            summary = render_prompt_dev_prompts(
                PromptDevRenderOptions(
                    classified_benchmark=classified,
                    world_state=world_state,
                    dev_manifest=manifest,
                    output_dir=root / "out",
                    max_cases=12,
                    sample_strategy="diverse_stratified",
                    representations=("hybrid_json_nl",),
                    example_policies=("zero_shot",),
                    context_bundles=("minimal_case",),
                    tasks=("track_diagnosis",),
                )
            )

            self.assertGreater(summary["counts"]["unique_qids"], 4)
            self.assertGreater(summary["counts"]["unique_properties"], 4)

    def test_prompt_dev_v3_templates_include_failure_taxonomy_rules(self) -> None:
        a_box = render_prompt_dev_prompt(
            task="a_box_repair",
            representation="hybrid_json_nl",
            case_payload={"id": "case_000001"},
        )
        t_box = render_prompt_dev_prompt(
            task="t_box_repair",
            representation="hybrid_json_nl",
            case_payload={"id": "case_000001"},
        )
        diagnosis = render_prompt_dev_prompt(
            task="track_diagnosis",
            representation="hybrid_json_nl",
            case_payload={"id": "case_000001"},
        )

        self.assertIn("Prompt version: prompt_dev_v3", a_box.user_prompt)
        self.assertIn("Replacement values must come from visible old-value normalization", a_box.user_prompt)
        self.assertIn("Preserve retained values", a_box.user_prompt)
        self.assertIn("targeted REMOVE", a_box.user_prompt)
        self.assertIn("Action decision tree", t_box.user_prompt)
        self.assertIn("compact_inventory_no_pre_change_signature", t_box.user_prompt)
        self.assertIn("Do not invent a full signature_after", t_box.user_prompt)
        self.assertIn("A constraint report alone does not imply T_BOX", diagnosis.user_prompt)
        self.assertIn("property-level schema-change evidence", diagnosis.user_prompt)


if __name__ == "__main__":
    unittest.main()
