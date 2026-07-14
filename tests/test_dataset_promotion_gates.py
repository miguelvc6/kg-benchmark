from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from artifact_lineage import validate_lineage
from classifier import lean_repair_target
from kg_benchmark.audit.workflow import _artifact
from kg_benchmark.dataset.gates import (
    AZURE_REQUESTED,
    MAIN_REQUESTED,
    RESERVE_REQUESTED,
    SCHEMA_ARTIFACTS,
    TBOX_TARGET,
    DatasetGateError,
    validate_dataset_semantics,
)
from kg_benchmark.dataset.release import promote_dataset, sha256_file, verify_dataset, write_source_provenance
from kg_benchmark.selection.extensible import (
    _effective_quotas,
    _support_group_keys,
    build_eligibility_order,
    build_support_bank,
    materialize_population,
    reserve_quotas,
    tbox_composition,
)
from kg_benchmark.selection.workflow import (
    _per_case_rows,
    _population_v2,
    _prompt_clean_order,
    _replacements,
)

ROOT = Path(__file__).resolve().parents[1]
NOW = "2026-07-14T00:00:00Z"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _popularity() -> dict:
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


def _base_record(index: int, stratum: str, tbox_index: int | None = None) -> tuple[dict, dict, dict]:
    qid = f"Q{index + 1}"
    pid = f"P{index + 1}"
    case_id = f"case-{index + 1:05d}"
    violation = {
        "report_page_title": f"Synthetic report {index}",
        "report_fix_date": "2026-01-01T00:00:00",
        "report_revision_old": index * 2 + 1,
        "report_revision_new": index * 2 + 2,
        "report_violation_type_raw": "Synthetic violation",
        "report_violation_type_normalized": "Synthetic violation",
        "report_violation_type_qids": [],
        "value": "old",
    }
    track = "T_BOX" if stratum == "TBOX" else "A_BOX"
    if track == "A_BOX":
        target = {
            "kind": "A_BOX",
            "author": "synthetic",
            "revision_id": index + 100,
            "action": "UPDATE",
            "old_value": "old",
            "new_value": "new",
        }
        classification = {
            "IC-L": ("TypeA", "TARGET_REQUIRED_CLAIM" if index % 2 == 0 else "FORMAT_NORMALIZATION"),
            "IC-G": ("TypeB", "LOCAL_TEXT_CONFIRMED"),
            "IC-E-elim": ("TypeC", "EXTERNAL_BY_ELIMINATION"),
        }[stratum]
        diagnostics = {}
    else:
        assert tbox_index is not None
        subtype = (
            "RELAXATION_SET_EXPANSION"
            if tbox_index < 170
            else "RESTRICTION_SET_CONTRACTION"
            if tbox_index < 250
            else "SCHEMA_UPDATE"
        )
        role = tbox_index % 4 if tbox_index < 16 else 3
        if role == 2:
            subtype = "COINCIDENTAL_SCHEMA_CHANGE"
        summary = {
            "target_constraint_qid": "Q21502404",
            "changed_qualifier_properties": ["P1793"] if role in {0, 1} else [],
            "added_values": ["new"] if role == 0 else [],
            "removed_values": ["old"] if role == 1 else [],
        }
        target = {
            "kind": "T_BOX",
            "author": "synthetic",
            "property_revision_id": index + 100,
            "property_revision_prev": index + 99,
            "constraint_delta": {"changed_constraint_types": ["Q21502404"]},
        }
        classification = ("T_BOX", subtype)
        diagnostics = {"tbox_diff_summary": summary}
    stage2 = {
        "id": case_id,
        "qid": qid,
        "property": pid,
        "track": track,
        "information_type": "TBD",
        "violation_context": violation,
        "repair_target": target,
        "persistence_check": {"status": "passed", "current_value_2026": "new"},
        "popularity": _popularity(),
    }
    stage4 = {
        **stage2,
        "repair_target": lean_repair_target(target),
        "labels_en": {
            "qid": {"label": f"Entity {index}", "description": "Synthetic entity"},
            "property": {"label": f"Property {index}", "description": "Synthetic property"},
        },
        "context_ref": {"world_state_id": case_id, "world_state_path": "source/world-state.jsonl"},
        "classification": {
            "class": classification[0],
            "subtype": classification[1],
            "confidence": "high",
            "decision_trace": [{"step": "synthetic", "result": "pass"}],
            "rationale": "Synthetic release-gate fixture.",
            "constraint_types": [],
            "diagnostics": diagnostics,
        },
        "build": {"fixture": True},
    }
    world = {
        "id": case_id,
        "world_state": {
            "L1_ego_node": {"qid": qid, "properties": {pid: ["new"]}},
            "L2_labels": {"entities": {}},
            "L3_neighborhood": {"outgoing_edges": []},
            "L4_constraints": {"property_id": pid, "constraints": []},
        },
    }
    return stage2, stage4, world


class DatasetPromotionGateTests(unittest.TestCase):
    def _candidate(self, root: Path) -> tuple[Path, Path, dict]:
        repo = root / "repo"
        work = repo / "work"
        (repo / "paper").mkdir(parents=True)
        (repo / "schemas").mkdir()
        for filename in SCHEMA_ARTIFACTS.values():
            shutil.copyfile(ROOT / "schemas" / filename, repo / "schemas" / filename)
        protocol = {
            "protocol_id": "synthetic-final-protocol",
            "status": "frozen",
            "conditions": {
                "prompt_regimes": ["zero_shot", "static_few_shot"],
                "context_bundles": ["logic_only", "local_graph"],
            },
        }
        _write_json(repo / "paper" / "protocol.json", protocol)
        lock = {
            "manifest_type": "paper_methodology_lock",
            "manifest_version": 1,
            "protocol_id": protocol["protocol_id"],
            "source_git_revision": "b" * 40,
            "freeze_scope_sha256": "a" * 64,
            "files": {"paper/protocol.json": sha256_file(repo / "paper" / "protocol.json")},
        }
        _write_json(repo / "paper" / "methodology.lock.json", lock)
        methodology = {
            "files": lock["files"],
            "freeze_scope_sha256": lock["freeze_scope_sha256"],
            "lock": {
                "path": "paper/methodology.lock.json",
                "source_git_revision": lock["source_git_revision"],
            },
        }

        counts = {"IC-L": 320, "IC-G": 500, "IC-E-elim": 400, "TBOX": 400}
        stage2_rows: list[dict] = []
        cases: list[dict] = []
        worlds: list[dict] = []
        tbox_index = 0
        for stratum, count in counts.items():
            for _ in range(count):
                stage2, stage4, world = _base_record(
                    len(cases),
                    stratum,
                    tbox_index if stratum == "TBOX" else None,
                )
                stage2_rows.append(stage2)
                cases.append(stage4)
                worlds.append(world)
                if stratum == "TBOX":
                    tbox_index += 1

        acquisition = work / "acquisition"
        stage0 = {row["qid"]: row["popularity"] for row in stage2_rows}
        stage1 = [
            {
                "qid": row["qid"],
                "property_id": row["property"],
                "violation_type": "Synthetic violation",
                "fix_date": row["violation_context"]["report_fix_date"],
                "report_revision_old": row["violation_context"]["report_revision_old"],
                "report_revision_new": row["violation_context"]["report_revision_new"],
            }
            for row in stage2_rows
        ]
        stage3 = {row["id"]: row["world_state"] for row in worlds}
        _write_json(acquisition / "00_entity_popularity.json", stage0)
        _write_json(acquisition / "01_repair_candidates.json", stage1)
        _write_json(acquisition / "02_wikidata_repairs.json", stage2_rows)
        _write_json(acquisition / "03_world_state.json", stage3)
        dump = work / "latest-all.json.gz"
        dump.write_bytes(b"synthetic wikidata dump")
        cache = work / "cache"
        cache.mkdir(parents=True)
        acquisition_config = {
            "manifest_type": "dataset_acquisition",
            "manifest_version": 1,
            "status": "complete",
            "started_at_utc": NOW,
            "completed_at_utc": NOW,
            "command": "acquire",
            "arguments": ["--refresh-candidates"],
            "methodology": {
                "freeze_scope_sha256": lock["freeze_scope_sha256"],
                "source_git_revision": lock["source_git_revision"],
                "methodology_lock_sha256": sha256_file(repo / "paper" / "methodology.lock.json"),
            },
        }
        _write_json(work / "acquisition-config.json", acquisition_config)

        _write_jsonl(work / "source" / "popularity.jsonl", [
            {"qid": qid, "popularity": payload} for qid, payload in stage0.items()
        ])
        _write_jsonl(work / "source" / "candidates.jsonl", stage1)
        _write_jsonl(work / "source" / "repairs.jsonl", stage2_rows)
        _write_jsonl(work / "source" / "world-state.jsonl", worlds)
        _write_jsonl(work / "cases.jsonl", cases)
        provenance_path = write_source_provenance(
            acquisition_dir=acquisition,
            work_dir=work,
            dump_path=dump,
            acquisition_config_path=work / "acquisition-config.json",
            cache_dir=cache,
        )
        source_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        lineage = validate_lineage(
            stage0_path=acquisition / "00_entity_popularity.json",
            stage1_path=acquisition / "01_repair_candidates.json",
            stage2_json_path=acquisition / "02_wikidata_repairs.json",
            stage2_jsonl_path=work / "source" / "repairs.jsonl",
            stage3_path=work / "source" / "world-state.jsonl",
            stage4_path=work / "cases.jsonl",
            source_provenance=[source_provenance],
        )
        self.assertTrue(lineage["validation"]["passed"])
        _write_json(work / "lineage.json", lineage)

        dispositions = [{"case_id": row["id"], "disposition": "include"} for row in cases]
        _write_jsonl(work / "audit" / "dispositions.jsonl", dispositions)
        def artifact(path: Path, records: int | None = None) -> dict:
            return _artifact(path, repo, records=records)
        summary = {
            "report_type": "canonical_audit_summary",
            "report_version": 1,
            "created_at_utc": NOW,
            "policy": {"selection_eligible_disposition": "include"},
            "counts": {
                "cases": len(cases),
                "rendered_prompts": len(cases) * 4,
                "construct_reviews": 450,
                "temporal_reviews": 50,
                "reviews": 500,
                "by_disposition": {"include": len(cases)},
            },
            "validation": {
                "complete_unique_disposition_coverage": True,
                "deterministic_gates_passed": True,
                "temporal_gate_passed": True,
                "review_complete": True,
                "unresolved_systemic_findings": 0,
                "selection_eligible_disposition": "include",
            },
            "reviewer": {"interface": "codex_cli", "model": "synthetic", "cli_version": "synthetic-1"},
            "provenance": {
                "protocol": artifact(repo / "paper" / "protocol.json"),
                "cases": artifact(work / "cases.jsonl", len(cases)),
                "world_state": artifact(work / "source" / "world-state.jsonl", len(worlds)),
                "stage2": artifact(work / "source" / "repairs.jsonl", len(stage2_rows)),
                "lineage_manifest": artifact(work / "lineage.json"),
                "stage4_schema": artifact(repo / "schemas" / "dataset-case.schema.json"),
                "construct_sample": artifact(work / "cases.jsonl", len(cases)),
                "rendered_prompts": artifact(work / "cases.jsonl", len(cases)),
                "render_summary": artifact(work / "lineage.json"),
                "deterministic_manifest": artifact(work / "lineage.json"),
                "review_schema": artifact(repo / "schemas" / "audit-summary.schema.json"),
                "review_run": artifact(work / "lineage.json"),
                "reviews": artifact(work / "cases.jsonl", len(cases)),
                "dispositions": artifact(work / "audit" / "dispositions.jsonl", len(dispositions)),
            },
        }
        _write_json(work / "audit" / "summary.json", summary)

        exclusions = {
            "manifest_type": "event_group_exclusions",
            "manifest_version": 1,
            "description": "No prior groups in the synthetic release fixture.",
            "group_keys": [],
        }
        _write_json(work / "selections" / "group-exclusions.json", exclusions)
        eligibility, records_by_id = build_eligibility_order(
            cases_path=work / "cases.jsonl",
            dispositions_path=work / "audit" / "dispositions.jsonl",
            seed=13,
            excluded_group_keys=set(),
            tbox_targets=TBOX_TARGET,
        )
        support = build_support_bank(
            eligibility_rows=eligibility,
            records_by_id=records_by_id,
            capacity_per_locus=16,
        )
        ranking = {
            "manifest_type": "selection_ranking",
            "manifest_version": 1,
            "algorithm": "sha256",
            "seed": 13,
            "group_units": {"A_BOX": "qid_property", "T_BOX": "property_revision"},
            "tbox_weighted_prefix_target": TBOX_TARGET,
            "tie_break": "case_id",
        }
        _write_json(work / "selections" / "ranking.json", ranking)
        _write_jsonl(work / "selections" / "eligibility-order.jsonl", eligibility)
        _write_json(work / "selections" / "support-bank.json", support)
        effective_reserve = reserve_quotas(
            requested=RESERVE_REQUESTED,
            eligibility_rows=eligibility,
            support_bank=support,
        )
        reserve = materialize_population(
            name="reserve-1440",
            quotas=effective_reserve,
            eligibility_rows=eligibility,
            support_bank=support,
        )
        reserve.update({
            "manifest_type": "selection_reserve",
            "manifest_version": 1,
            "requested_quotas": RESERVE_REQUESTED,
            "tbox_composition": tbox_composition(eligibility, reserve["selected_case_ids"]),
        })
        reserve.pop("parent")
        _write_json(work / "selections" / "reserve.json", reserve)
        reserve_ids = set(reserve["selected_case_ids"])
        reserve_rows = [row for row in eligibility if row["case_id"] in reserve_ids]
        clean = _prompt_clean_order(reserve_rows, {}, TBOX_TARGET)
        per_case = _per_case_rows(
            cases_path=work / "cases.jsonl",
            dispositions_path=work / "audit" / "dispositions.jsonl",
            eligibility_rows=eligibility,
            support_bank=support,
            excluded_groups=set(),
            reserve_ids=reserve_ids,
            failure_reasons={},
        )
        _write_jsonl(work / "selections" / "per-case-eligibility.jsonl", per_case)
        _write_jsonl(work / "selections" / "prompt-clean-eligibility-order.jsonl", clean)
        prompt_audit = {
            "report_type": "selection_prompt_audit",
            "report_version": 1,
            "seed": 13,
            "reserve_cases": reserve["case_count"],
            "rendered_prompts": reserve["case_count"] * 8,
            "deterministically_scanned_prompts": reserve["case_count"] * 8,
            "temporal_review_cases": 50,
            "failed_reserve_cases": 0,
            "prompt_clean_reserve_cases": reserve["case_count"],
            "replacement_count": 0,
            "failed_case_ids": [],
            "failure_reasons": {},
            "validation": {
                "all_reserve_prompts_attempted": True,
                "all_rendered_prompts_scanned": True,
                "fixed_temporal_review_complete": True,
                "all_main_cases_prompt_clean": True,
                "all_azure_cases_prompt_clean": True,
            },
            "reviewer": {"interface": "codex_cli", "model": "synthetic", "cli_version": "synthetic-1"},
        }
        _write_json(work / "selections" / "prompt-audit.json", prompt_audit)
        effective_main = _effective_quotas(requested=MAIN_REQUESTED, eligibility_rows=clean, support_bank=support)
        main = materialize_population(
            name="main-1200",
            quotas=effective_main,
            eligibility_rows=clean,
            support_bank=support,
        )
        main_ids = set(main["selected_case_ids"])
        main_order = [row for row in clean if row["case_id"] in main_ids]
        effective_azure = _effective_quotas(
            requested=AZURE_REQUESTED,
            eligibility_rows=main_order,
            support_bank=support,
        )
        azure = materialize_population(
            name="azure-600",
            quotas=effective_azure,
            eligibility_rows=main_order,
            support_bank=support,
        )
        eligibility_by_case = {row["case_id"]: row for row in per_case}
        provenance = {
            "dataset": artifact(work / "cases.jsonl", len(cases)),
            "audit": artifact(work / "audit" / "summary.json"),
            "audit_dispositions": artifact(work / "audit" / "dispositions.jsonl", len(dispositions)),
            "exclusions": artifact(work / "selections" / "group-exclusions.json"),
            "ranking": artifact(work / "selections" / "ranking.json"),
            "support_bank": artifact(work / "selections" / "support-bank.json"),
            "eligibility_order": artifact(work / "selections" / "eligibility-order.jsonl", len(eligibility)),
            "prompt_audit": artifact(work / "selections" / "prompt-audit.json"),
            "per_case_eligibility": artifact(work / "selections" / "per-case-eligibility.jsonl", len(per_case)),
        }
        main = _population_v2(
            population=main,
            requested_quotas=MAIN_REQUESTED,
            ordering_rows=clean,
            provenance=provenance,
            eligibility_by_case=eligibility_by_case,
            excluded_groups=set(),
            support_groups=_support_group_keys(support),
        )
        azure = _population_v2(
            population=azure,
            requested_quotas=AZURE_REQUESTED,
            ordering_rows=main_order,
            provenance=provenance,
            eligibility_by_case=eligibility_by_case,
            excluded_groups=set(),
            support_groups=_support_group_keys(support),
            parent={"name": "main-1200", "nesting_proven": True},
            parent_case_ids=main_ids,
            parent_relationship="subset_of_parent",
        )
        _write_json(work / "selections" / "main-1200.json", main)
        _write_json(work / "selections" / "azure-600.json", azure)
        replacements = _replacements(reserve_rows, set(row["case_id"] for row in clean), effective_main)
        _write_jsonl(work / "selections" / "replacements.jsonl", replacements)
        return repo, work, methodology

    def test_complete_candidate_promotes_and_reproduces_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo, work, methodology = self._candidate(Path(temporary))
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
            result = verify_dataset(dataset)
            self.assertEqual(manifest["manifest_version"], 2)
            self.assertTrue(result["manifest_byte_reproduced"])
            self.assertEqual(result["semantic_validation"]["selection"]["main_cases"], 1200)
            self.assertEqual(result["semantic_validation"]["selection"]["azure_cases"], 600)
            (dataset / "source" / "candidates.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetGateError, "schema_candidate"):
                validate_dataset_semantics(dataset)

    def test_gate_rejects_incomplete_dispositions_without_partial_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo, work, methodology = self._candidate(Path(temporary))
            disposition_path = work / "audit" / "dispositions.jsonl"
            disposition_path.write_text(
                "\n".join(disposition_path.read_text(encoding="utf-8").splitlines()[:-1]) + "\n",
                encoding="utf-8",
            )
            dataset = repo / "dataset"
            with self.assertRaisesRegex(DatasetGateError, "dispositions"):
                promote_dataset(
                    work_dir=work,
                    dataset_dir=dataset,
                    protocol_path=repo / "paper" / "protocol.json",
                    source_provenance_path=work / "source-provenance.json",
                    repo_root=repo,
                    methodology_check=lambda _: methodology,
                )
            self.assertFalse(dataset.exists())

    def test_lineage_schema_and_population_mutations_fail_closed(self) -> None:
        mutations = {
            "lineage": (
                self._invalidate_lineage,
                "lineage manifest",
            ),
            "source_provenance": (
                self._alter_original_dump,
                "Source-provenance artifact wikidata_dump",
            ),
            "record_schema": (
                lambda work: (work / "source" / "candidates.jsonl").write_text("{}\n", encoding="utf-8"),
                "canonical candidates",
            ),
            "canonical_source_equivalence": (
                self._alter_canonical_popularity,
                "canonical popularity",
            ),
            "population_size": (
                lambda work: self._change_main_count(work, 1199),
                "exactly 1,200",
            ),
            "independent_groups": (
                self._duplicate_main_group,
                "non-unique elements",
            ),
            "prior_group_exclusion": (
                self._exclude_first_eligible_group,
                "eligibility ordering",
            ),
            "support_exclusion": (
                self._alter_support_group,
                "support bank",
            ),
            "reserve_finalization": (
                self._alter_reserve_case,
                "reserve",
            ),
            "prompt_qa": (
                self._alter_prompt_scan_count,
                "deterministic scanning",
            ),
            "systemic_finding": (
                self._add_systemic_finding,
                "schema_audit_summary",
            ),
            "azure_nesting": (
                self._break_azure_nesting,
                "nested 600-case subset",
            ),
        }
        for name, (mutate, message) in mutations.items():
            with self.subTest(gate=name), tempfile.TemporaryDirectory() as temporary:
                repo, work, methodology = self._candidate(Path(temporary))
                mutate(work)
                dataset = repo / "dataset"
                with self.assertRaisesRegex((DatasetGateError, ValueError), message):
                    promote_dataset(
                        work_dir=work,
                        dataset_dir=dataset,
                        protocol_path=repo / "paper" / "protocol.json",
                        source_provenance_path=work / "source-provenance.json",
                        repo_root=repo,
                        methodology_check=lambda _: methodology,
                    )
                self.assertFalse(dataset.exists())

    @staticmethod
    def _change_main_count(work: Path, count: int) -> None:
        path = work / "selections" / "main-1200.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["case_count"] = count
        _write_json(path, value)

    @staticmethod
    def _invalidate_lineage(work: Path) -> None:
        path = work / "lineage.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["validation"]["passed"] = False
        _write_json(path, value)

    @staticmethod
    def _duplicate_main_group(work: Path) -> None:
        path = work / "selections" / "main-1200.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["selected_group_keys"][1] = value["selected_group_keys"][0]
        _write_json(path, value)

    @staticmethod
    def _exclude_first_eligible_group(work: Path) -> None:
        eligibility = json.loads(
            (work / "selections" / "eligibility-order.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        path = work / "selections" / "group-exclusions.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["group_keys"] = [eligibility["group_key"]]
        _write_json(path, value)

    @staticmethod
    def _alter_support_group(work: Path) -> None:
        path = work / "selections" / "support-bank.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["support_sets"]["a_box_repair"][0]["group_key"] = "ABOX|Q999999999|P999999999"
        _write_json(path, value)

    @staticmethod
    def _alter_reserve_case(work: Path) -> None:
        path = work / "selections" / "reserve.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["selected_case_ids"][0] = "not-a-real-case"
        _write_json(path, value)

    @staticmethod
    def _alter_prompt_scan_count(work: Path) -> None:
        path = work / "selections" / "prompt-audit.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["deterministically_scanned_prompts"] -= 1
        _write_json(path, value)

    @staticmethod
    def _alter_original_dump(work: Path) -> None:
        provenance = json.loads((work / "source-provenance.json").read_text(encoding="utf-8"))
        Path(provenance["sources"]["wikidata_dump"]["path"]).write_bytes(b"changed dump bytes")

    @staticmethod
    def _alter_canonical_popularity(work: Path) -> None:
        path = work / "source" / "popularity.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["popularity"]["score"] = 0.75
        _write_jsonl(path, rows)

    @staticmethod
    def _add_systemic_finding(work: Path) -> None:
        path = work / "audit" / "summary.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["validation"]["unresolved_systemic_findings"] = 1
        _write_json(path, value)

    @staticmethod
    def _break_azure_nesting(work: Path) -> None:
        main = json.loads((work / "selections" / "main-1200.json").read_text(encoding="utf-8"))
        azure_path = work / "selections" / "azure-600.json"
        azure = json.loads(azure_path.read_text(encoding="utf-8"))
        clean = [
            json.loads(line)
            for line in (work / "selections" / "prompt-clean-eligibility-order.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        replacement = next(row for row in clean if row["case_id"] not in set(main["selected_case_ids"]))
        eligibility = {
            row["case_id"]: row
            for row in (
                json.loads(line)
                for line in (work / "selections" / "per-case-eligibility.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
        }
        removed = azure["selected_case_ids"][0]
        azure["selected_case_ids"][0] = replacement["case_id"]
        azure["selected_group_keys"][0] = replacement["group_key"]
        azure["case_eligibility_sha256"].pop(removed)
        azure["case_eligibility_sha256"][replacement["case_id"]] = eligibility[replacement["case_id"]][
            "eligibility_sha256"
        ]
        _write_json(azure_path, azure)

    def test_verify_rejects_nonreproducible_manifest_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo, work, methodology = self._candidate(Path(temporary))
            dataset = repo / "dataset"
            promote_dataset(
                work_dir=work,
                dataset_dir=dataset,
                protocol_path=repo / "paper" / "protocol.json",
                source_provenance_path=work / "source-provenance.json",
                repo_root=repo,
                methodology_check=lambda _: methodology,
            )
            manifest_path = dataset / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "byte-reproducible"):
                verify_dataset(dataset)


if __name__ == "__main__":
    unittest.main()
