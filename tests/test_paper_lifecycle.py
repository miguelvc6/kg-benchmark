from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from guardian.generation_cache import GenerationCache
from kg_benchmark.analysis.workflow import (
    build_paper_results,
    replay_matrix_evaluations,
    verify_paper_results,
)
from kg_benchmark.audit.workflow import (
    audit_status,
    prepare_audit,
    run_deterministic_phase,
    run_finalize_phase,
    run_review_phase,
)
from kg_benchmark.cli import _run_build
from kg_benchmark.dataset.release import promote_dataset, sha256_file, verify_dataset
from kg_benchmark.matrix.workflow import dry_run_matrix, execute_matrix, matrix_status, plan_matrix
from kg_benchmark.selection.extensible import group_key_for_record, stratum_for_record
from kg_benchmark.selection.workflow import (
    finalize_selection,
    prepare_reserve,
    review_reserve,
    selection_status,
)

ROOT = Path(__file__).resolve().parents[1]
NOW = "2026-07-14T00:00:00Z"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _popularity() -> dict[str, Any]:
    return {
        "score": 0.5,
        "components": {
            "pageviews_365d": 1,
            "out_degree": 1,
            "sitelinks_count": 1,
            "pageviews_norm": 0.5,
            "degree_norm": 0.5,
            "sitelinks_norm": 0.5,
        },
    }


def _signature(values: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entries = [
        {
            "constraint_qid": "Q21510859",
            "constraint_label": "one-of constraint",
            "qualifiers": [
                {"property_id": "P2305", "values": [{"raw": value} for value in values]}
            ],
        }
    ]
    raw = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return entries, {
        "signature_raw": raw,
        "hash": hashlib.sha1(raw.encode()).hexdigest(),
        "signature": entries,
    }


def _source_case(index: int, stratum: str, tbox_index: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    case_id = f"case-{index:05d}"
    qid = f"Q{100000 + index}"
    pid = f"P{200000 + index}"
    violation_name = "Item"
    old_value: Any = f"old-{index}"
    new_value: Any = f"Q{600000 + index}"
    action = "UPDATE"
    track = "A_BOX"
    target: dict[str, Any]
    world = {
        "L1_ego_node": {
            "qid": qid,
            "label": f"Unrelated entity {index}",
            "description": f"Unrelated description {index}",
            "properties": {pid: [new_value], "P999": [f"Q{800000 + index}"]},
        },
        "L2_labels": {"entities": {f"Q{800000 + index}": {"label": f"Neighbor {index}"}}},
        "L3_neighborhood": {"outgoing_edges": []},
        "L4_constraints": {"property_id": pid, "constraints": []},
    }
    if stratum == "IC-L":
        if index % 2 == 0:
            violation_name = f"Target required claim P|{pid[1:]}"
            old_value = "MISSING"
            new_value = qid
            action = "CREATE"
        else:
            violation_name = "Format"
            old_value = f"CODE{index}/"
            new_value = f"CODE{index}"
            world["L1_ego_node"]["properties"][pid] = [new_value]
            world["L4_constraints"]["constraints"] = [
                {
                    "constraint_type": {"qid": "Q21502404", "label": "format constraint"},
                    "qualifiers": [{"property_id": "P1793", "values": [{"raw": "[A-Z]+[0-9]+"}]}],
                }
            ]
    elif stratum == "IC-G":
        new_value = f"local-{index}"
        world["L1_ego_node"]["properties"] = {pid: [new_value], "P999": [new_value]}
    elif stratum == "TBOX":
        assert tbox_index is not None
        track = "T_BOX"
        violation_name = (
            "Format" if tbox_index < 4 else "Unmapped fixture violation" if tbox_index < 8 else "One of"
        )
        base_value = f"Q{300000 + index}"
        changed_value = f"Q{400000 + index}"
        replacement_value = f"Q{500000 + index}"
        if tbox_index < 170:
            before_values, after_values = [base_value], [base_value, changed_value]
        elif tbox_index < 250:
            before_values, after_values = [base_value, changed_value], [base_value]
        else:
            before_values, after_values = [base_value], [replacement_value]
        before_entries, before_signature = _signature(before_values)
        after_entries, after_signature = _signature(after_values)
        target = {
            "kind": "T_BOX",
            "author": None,
            "property_revision_id": 900000 + index,
            "property_revision_prev": 899999 + index,
            "constraint_delta": {
                "changed_constraint_types": [],
                "signature_before": before_entries,
                "signature_after": after_entries,
                "hash_before": before_signature["hash"],
                "hash_after": after_signature["hash"],
            },
        }
        world["constraint_change_context"] = {
            "signatures": {"before": before_signature, "after": after_signature}
        }
        old_value, new_value = before_values, after_values
    if track == "A_BOX":
        target = {
            "kind": "A_BOX",
            "author": None,
            "revision_id": 700000 + index,
            "action": action,
            "old_value": old_value,
            "new_value": new_value,
        }
    violation = {
        "report_page_title": f"Fixture report {index}",
        "report_fix_date": "2026-01-01T00:00:00",
        "report_revision_old": index * 2 + 1,
        "report_revision_new": index * 2 + 2,
        "report_violation_type_raw": violation_name,
        "report_violation_type_normalized": violation_name,
        "report_violation_type_qids": [],
        "value": old_value,
    }
    record = {
        "id": case_id,
        "qid": qid,
        "qid_label_en": f"Fixture entity {index}",
        "qid_description_en": f"Fixture entity description {index}",
        "property": pid,
        "property_label_en": f"Fixture property {index}",
        "property_description_en": f"Fixture property description {index}",
        "track": track,
        "information_type": "fixture",
        "violation_context": violation,
        "repair_target": target,
        "persistence_check": {"status": "passed", "current_value_2026": new_value},
        "popularity": _popularity(),
    }
    return record, world


def _write_acquisition(root: Path, lock_sha256: str, revision: str, freeze_sha256: str) -> None:
    acquisition = root / "work" / "acquisition"
    rows: list[dict[str, Any]] = []
    worlds: dict[str, dict[str, Any]] = {}
    index = 1
    counts = {"IC-L": 320, "IC-G": 500, "IC-E-elim": 400, "TBOX": 400}
    tbox_index = 0
    for stratum, count in counts.items():
        for _ in range(count):
            record, world = _source_case(index, stratum, tbox_index if stratum == "TBOX" else None)
            rows.append(record)
            worlds[record["id"]] = world
            index += 1
            if stratum == "TBOX":
                tbox_index += 1
    stage0 = {row["qid"]: row["popularity"] for row in rows}
    stage1 = [
        {
            "qid": row["qid"],
            "property_id": row["property"],
            "violation_type": row["violation_context"]["report_violation_type_normalized"],
            "fix_date": row["violation_context"]["report_fix_date"],
            "report_revision_old": row["violation_context"]["report_revision_old"],
            "report_revision_new": row["violation_context"]["report_revision_new"],
            "report_event_sampling": {
                "method": "sha256_qid_rank_v1",
                "seed": 13,
                "cap": 100,
                "event_key_sha256": f"{int(row['qid'][1:]):064x}",
                "event_candidate_count": 1,
                "selected_candidate_count": 1,
                "rank": 1,
                "capped": False,
            },
        }
        for row in rows
    ]
    _write_json(acquisition / "00_entity_popularity.json", stage0)
    _write_json(acquisition / "01_repair_candidates.json", stage1)
    _write_json(acquisition / "02_wikidata_repairs.json", rows)
    _write_json(acquisition / "02_stage2_exclusions.json", [])
    _write_json(acquisition / "03_world_state.json", worlds)
    dump = root / "work" / "latest-all.json.gz"
    dump.write_bytes(b"synthetic wikidata dump\n")
    (root / "work" / "cache").mkdir(parents=True)
    _write_json(
        root / "work" / "acquisition-config.json",
        {
            "manifest_type": "dataset_acquisition",
            "manifest_version": 1,
            "status": "complete",
            "started_at_utc": NOW,
            "completed_at_utc": NOW,
            "command": "acquire",
            "arguments": ["--fixture"],
            "methodology": {
                "freeze_scope_sha256": freeze_sha256,
                "source_git_revision": revision,
                "methodology_lock_sha256": lock_sha256,
            },
        },
    )


class FakeCodex:
    def __init__(self, fail_packet_id: str | None = None) -> None:
        self.fail_packet_id = fail_packet_id

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command == ["codex", "--version"]:
            return subprocess.CompletedProcess(command, 0, "codex-cli lifecycle-test-1.0\n", "")
        packets = json.loads(kwargs["input"].split("\n\n", 1)[1])
        reviews = [
            {
                "packet_id": packet["packet_id"],
                "audit_dimension": packet.get("audit_dimension", "temporal"),
                "verdict": "uncertain" if packet["packet_id"] == self.fail_packet_id else "pass",
                "rationale": "Offline synthetic acceptance review.",
                "evidence": ["blinded fixture packet"],
            }
            for packet in packets
        ]
        return subprocess.CompletedProcess(command, 0, json.dumps({"reviews": reviews}), "")


class PaperLifecycleAcceptanceTests(unittest.TestCase):
    def _repository(self, root: Path) -> tuple[Path, dict[str, Any]]:
        repo = root / "repo"
        shutil.copytree(ROOT / "paper", repo / "paper")
        shutil.copytree(ROOT / "schemas", repo / "schemas")
        protocol_path = repo / "paper" / "protocol.json"
        policy_path = repo / "paper" / "selection-policy.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["status"] = "frozen"
        _write_json(protocol_path, protocol)
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        policy["status"] = "frozen"
        _write_json(policy_path, policy)
        model = {
            "model_id": "ollama_fixture",
            "provider": "ollama",
            "model": "fixture:1b",
            "model_revision": "d" * 64,
            "revision_status": "resolved",
            "revision_resolution": "synthetic acceptance fixture",
            "population": "mini-4",
            "execution_mode": "sync",
            "temperature": 0,
            "top_p": 1,
            "seed": 13,
            "context_length": 4096,
            "max_output_tokens": 512,
            "ollama_think": "disabled",
            "max_transport_retries": 2,
            "tools_disabled": True,
            "expected_calls": 32,
        }
        models = {
            "manifest_type": "paper_model_matrix",
            "manifest_version": 1,
            "status": "frozen",
            "request_dimensions": {
                "tasks": ["repair_proposal", "track_diagnosis"],
                "prompt_regimes": ["zero_shot", "static_few_shot"],
                "context_bundles": ["logic_only", "local_graph"],
            },
            "generation_identity": ["prompt", "case", "context", "model", "revision", "inference"],
            "models": [model],
        }
        models_path = repo / "paper" / "models.json"
        _write_json(models_path, models)
        freeze_sha256 = "f" * 64
        revision = "b" * 40
        lock = {
            "manifest_type": "paper_methodology_lock",
            "manifest_version": 1,
            "protocol_id": protocol["protocol_id"],
            "source_git_revision": revision,
            "freeze_scope_sha256": freeze_sha256,
            "files": {
                "paper/models.json": sha256_file(models_path),
                "paper/protocol.json": sha256_file(protocol_path),
            },
        }
        lock_path = repo / "paper" / "methodology.lock.json"
        _write_json(lock_path, lock)
        _write_acquisition(repo, sha256_file(lock_path), revision, freeze_sha256)
        methodology = {
            "files": lock["files"],
            "freeze_scope_sha256": freeze_sha256,
            "lock": {"path": "paper/methodology.lock.json", "source_git_revision": revision},
        }
        return repo, methodology

    def test_complete_offline_paper_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, methodology = self._repository(root)
            work = repo / "work"

            with patch("kg_benchmark.cli.require_frozen_methodology", return_value=methodology):
                with contextlib.redirect_stdout(None):
                    self.assertEqual(
                        _run_build(
                            [
                                "--work-dir", str(work),
                                "--acquisition-dir", str(work / "acquisition"),
                                "--dump-path", str(work / "latest-all.json.gz"),
                                "--cache-dir", str(work / "cache"),
                                "--acquisition-config", str(work / "acquisition-config.json"),
                                "--quiet",
                                "--no-progress",
                            ]
                        ),
                        0,
                    )
            lineage = json.loads((work / "lineage.json").read_text(encoding="utf-8"))
            self.assertTrue(lineage["validation"]["passed"])
            built_cases = [
                json.loads(line) for line in (work / "cases.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                {row["context_ref"]["world_state_path"] for row in built_cases},
                {"source/world-state.jsonl"},
            )

            audit_dir = work / "audit"
            report_path = repo / "audit.md"
            prepare_audit(
                cases_path=work / "cases.jsonl",
                world_state_path=work / "source" / "world-state.jsonl",
                stage2_path=work / "source" / "repairs.jsonl",
                lineage_manifest_path=work / "lineage.json",
                stage4_schema_path=repo / "schemas" / "dataset-case.schema.json",
                protocol_path=repo / "paper" / "protocol.json",
                work_dir=audit_dir,
                repo_root=repo,
            )
            deterministic = run_deterministic_phase(work_dir=audit_dir, repo_root=repo)
            self.assertTrue(deterministic["phases"]["deterministic"]["passed"])
            run_review_phase(work_dir=audit_dir, repo_root=repo, run_command=FakeCodex())
            run_finalize_phase(work_dir=audit_dir, report_path=report_path, repo_root=repo)
            self.assertEqual(audit_status(work_dir=audit_dir, repo_root=repo)["valid_through"], "finalize")

            exclusions = work / "prior-groups.json"
            _write_json(
                exclusions,
                {
                    "manifest_type": "event_group_exclusions",
                    "manifest_version": 1,
                    "description": "No prior groups in the offline lifecycle fixture.",
                    "group_keys": [],
                },
            )
            selection_dir = work / "selections"
            prepare_reserve(
                cases_path=work / "cases.jsonl",
                world_state_path=work / "source" / "world-state.jsonl",
                dispositions_path=audit_dir / "dispositions.jsonl",
                audit_summary_path=audit_dir / "summary.json",
                exclusions_path=exclusions,
                protocol_path=repo / "paper" / "protocol.json",
                policy_path=repo / "paper" / "selection-policy.json",
                output_dir=selection_dir,
                repo_root=repo,
            )
            reserve = json.loads((selection_dir / "reserve.json").read_text(encoding="utf-8"))
            initial_main: set[str] = set()
            offset = 0
            main_quotas = {"IC-L": 230, "IC-G": 375, "IC-E-elim": 295, "TBOX": 300}
            for stratum in ("IC-L", "IC-G", "IC-E-elim", "TBOX"):
                reserve_count = reserve["quotas"][stratum]
                initial_main.update(
                    reserve["selected_case_ids"][offset : offset + main_quotas[stratum]]
                )
                offset += reserve_count
            private_map = json.loads(
                (selection_dir / "private-temporal-review-map.json").read_text(encoding="utf-8")
            )
            failed_packet = next(
                packet_id for packet_id, case_id in private_map.items() if case_id in initial_main
            )
            review_reserve(
                output_dir=selection_dir,
                repo_root=repo,
                run_command=FakeCodex(fail_packet_id=failed_packet),
            )
            finalized = finalize_selection(output_dir=selection_dir, repo_root=repo)
            counts = finalized["phases"]["finalize"]["counts"]
            self.assertEqual((counts["main_cases"], counts["azure_cases"]), (1200, 600))
            self.assertGreater(counts["replacements"], 0)
            self.assertEqual(selection_status(output_dir=selection_dir, repo_root=repo)["valid_through"], "finalize")

            dataset = repo / "dataset"
            manifest = promote_dataset(
                work_dir=work,
                dataset_dir=dataset,
                protocol_path=repo / "paper" / "protocol.json",
                source_provenance_path=work / "source-provenance.json",
                lineage_manifest_path=work / "lineage.json",
                repo_root=repo,
                methodology_check=lambda _: methodology,
            )
            verified_dataset = verify_dataset(dataset, check_external_sources=True)
            self.assertEqual(manifest["manifest_version"], 2)
            self.assertTrue(verified_dataset["manifest_byte_reproduced"])

            cases = {
                row["id"]: row
                for row in (
                    json.loads(line) for line in (dataset / "cases.jsonl").read_text(encoding="utf-8").splitlines()
                )
            }
            main = json.loads((dataset / "selections" / "main-1200.json").read_text(encoding="utf-8"))
            chosen: list[str] = []
            for stratum in ("IC-L", "IC-G", "IC-E-elim", "TBOX"):
                chosen.append(next(case_id for case_id in main["selected_case_ids"] if stratum_for_record(cases[case_id]) == stratum))
            mini = {
                "manifest_type": "evaluation_population",
                "manifest_version": 1,
                "name": "mini-4",
                "quotas": {"IC-L": 1, "IC-G": 1, "IC-E-elim": 1, "TBOX": 1},
                "case_count": 4,
                "selected_case_ids": chosen,
                "selected_group_keys": [group_key_for_record(cases[case_id]) for case_id in chosen],
                "support_groups_excluded": 32,
                "parent": None,
            }
            mini_path = repo / "mini-4.json"
            _write_json(mini_path, mini)
            matrix_result = plan_matrix(
                dataset_dir=dataset,
                models_path=repo / "paper" / "models.json",
                protocol_path=repo / "paper" / "protocol.json",
                output_root=repo / "runs" / "matrices",
                population_paths=[mini_path],
                schema_root=repo / "schemas",
            )
            matrix_dir = Path(matrix_result["matrix_dir"])
            matrix_bytes = (matrix_dir / "matrix.json").read_bytes()
            repeated_plan = plan_matrix(
                dataset_dir=dataset,
                models_path=repo / "paper" / "models.json",
                protocol_path=repo / "paper" / "protocol.json",
                output_root=repo / "runs" / "matrices",
                population_paths=[mini_path],
                schema_root=repo / "schemas",
            )
            self.assertEqual(Path(repeated_plan["matrix_dir"]), matrix_dir)
            self.assertEqual((matrix_dir / "matrix.json").read_bytes(), matrix_bytes)
            cache_path = repo / "runs" / "generation-cache.sqlite"
            self.assertEqual(dry_run_matrix(matrix_dir=matrix_dir, generation_cache_path=cache_path, schema_root=repo / "schemas")["unique_new_requests"], 32)
            requests = [json.loads(line) for line in (matrix_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()]

            def fake_executor(**kwargs: Any) -> dict[str, Any]:
                group_dir = Path(kwargs["resume_run_dir"])
                population = json.loads(Path(kwargs["selection_manifest_path"]).read_text(encoding="utf-8"))
                context = kwargs["ablation_bundles"][0]
                regime = kwargs["prompt_regime"]
                selected = [
                    row
                    for row in requests
                    if row["population"] == population["name"]
                    and row["context_bundle"] == context
                    and row["prompt_regime"] == regime
                ]
                cache = GenerationCache(Path(kwargs["generation_cache_path"]))
                for row in selected:
                    cache.put(row["request_spec"], raw_response={}, parsed_payload={}, usage={})
                run_rows = [
                    {
                        "case_id": row["case_id"],
                        "task_type": "proposal" if row["task"] == "repair_proposal" else "track_diagnosis",
                        "ablation_bundle": context,
                        "parse_status": "normalized",
                    }
                    for row in selected
                ]
                _write_json(
                    group_dir / "run_config.json",
                    {
                        "classified_benchmark": str(dataset / "cases.jsonl"),
                        "world_state": str(dataset / "source" / "world-state.jsonl"),
                        "selection_manifest": str(mini_path),
                        "selected_case_ids": population["selected_case_ids"],
                        "ablation_bundles": [context],
                        "tbox_task_version": "tbox_taxonomy_patch_v1",
                        "prompt_regime": regime,
                    },
                )
                _write_jsonl(group_dir / "run_manifest.jsonl", run_rows)
                _write_json(group_dir / "reasoning_floor_summary.json", {"requests": len(run_rows)})
                bundle = group_dir / context
                for filename in (
                    "a_box_proposals.jsonl",
                    "t_box_proposals.jsonl",
                    "t_box_taxonomy_patch_proposals.jsonl",
                    "track_diagnoses.jsonl",
                ):
                    _write_jsonl(bundle / filename, [])
                return {"provider_calls": 0}

            executed = execute_matrix(
                matrix_dir=matrix_dir,
                generation_cache_path=cache_path,
                repo_root=repo,
                schema_root=repo / "schemas",
                methodology_check=lambda _: methodology,
                run_executor=fake_executor,
            )
            self.assertTrue(executed["complete"])
            self.assertEqual(executed["executed_groups"], 4)
            cached = dry_run_matrix(matrix_dir=matrix_dir, generation_cache_path=cache_path, schema_root=repo / "schemas")
            self.assertEqual((cached["unique_new_requests"], cached["unique_cache_hits"]), (0, 32))
            resumed = execute_matrix(
                matrix_dir=matrix_dir,
                generation_cache_path=cache_path,
                repo_root=repo,
                schema_root=repo / "schemas",
                methodology_check=lambda _: methodology,
                run_executor=fake_executor,
            )
            self.assertEqual((resumed["executed_groups"], resumed["skipped_groups"]), (0, 4))
            self.assertTrue(matrix_status(matrix_dir=matrix_dir, generation_cache_path=cache_path, schema_root=repo / "schemas")["complete"])

            replay = replay_matrix_evaluations(
                matrix_dir=matrix_dir,
                evaluation_id="lifecycle-v1",
                generation_cache_path=cache_path,
                schema_root=repo / "schemas",
            )
            self.assertEqual((replay["provider_calls"], replay["created_evaluations"]), (0, 4))
            analysis = json.loads((repo / "paper" / "analysis.json").read_text(encoding="utf-8"))
            analysis.update(
                {
                    "status": "frozen",
                    "confirmatory_population": "mini-4",
                    "calibration_population": "api-mini",
                }
            )
            analysis_path = repo / "paper" / "analysis.json"
            _write_json(analysis_path, analysis)
            result = build_paper_results(
                matrix_dir=matrix_dir,
                evaluation_id="lifecycle-v1",
                generation_cache_path=cache_path,
                analysis_config_path=analysis_path,
                output_root=repo / "results-a",
                schema_root=repo / "schemas",
                repo_root=ROOT,
            )
            result_dir = Path(result["result_dir"])
            self.assertTrue(verify_paper_results(result_dir=result_dir, schema_root=repo / "schemas")["valid"])
            reproduced = build_paper_results(
                matrix_dir=matrix_dir,
                evaluation_id="lifecycle-v1",
                generation_cache_path=cache_path,
                analysis_config_path=analysis_path,
                output_root=repo / "results-b",
                schema_root=repo / "schemas",
                repo_root=ROOT,
            )
            reproduced_dir = Path(reproduced["result_dir"])
            for relative in ("manifest.json", "summary.json", "estimates.jsonl", "contrasts.jsonl"):
                self.assertEqual((result_dir / relative).read_bytes(), (reproduced_dir / relative).read_bytes())


if __name__ == "__main__":
    unittest.main()
