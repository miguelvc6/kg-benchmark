import json
import tempfile
import unittest
from pathlib import Path

from kg_benchmark.selection.extensible import (
    DEFAULT_API_QUOTAS,
    DEFAULT_MAIN_QUOTAS,
    DEFAULT_TBOX_TARGET,
    _weighted_tbox_order,
    build_selection_artifacts,
    materialize_population,
)


def _abox(case_id: str, index: int, class_name: str, subtype: str) -> dict:
    return {
        "id": case_id,
        "qid": f"Q{index}",
        "property": f"P{index}",
        "track": "A_BOX",
        "classification": {"class": class_name, "subtype": subtype},
    }


def _tbox(case_id: str, index: int, role_index: int) -> dict:
    gold_by_role = (
        {"schema_decision": "CAUSAL_SCHEMA_REPAIR", "repairs": [{"taxonomy_code": "CQ_PLUS"}]},
        {"schema_decision": "CAUSAL_SCHEMA_REPAIR", "repairs": [{"taxonomy_code": "CQ_MINUS"}]},
        {"schema_decision": "NO_CAUSAL_SCHEMA_REPAIR", "repairs": []},
        {"schema_decision": "UNCLEAR_SCHEMA_EVIDENCE", "repairs": [{"taxonomy_code": "OTHER"}]},
    )
    return {
        "id": case_id,
        "qid": f"Q{index}",
        "property": f"P{index}",
        "track": "T_BOX",
        "classification": {"class": "T_BOX", "subtype": "SCHEMA_UPDATE"},
        "repair_target": {"property_revision_id": index},
        "gold": gold_by_role[role_index % 4],
    }


class ExtensibleSelectionTests(unittest.TestCase):
    def test_weighted_tbox_prefix_reaches_declared_main_target(self) -> None:
        rows = [
            {
                "case_id": f"{category}-{index}",
                "group_key": f"TBOX|P{category}|{index}",
                "stratum": "TBOX",
                "tbox_category": category,
                "rank": f"{index:064x}",
            }
            for category, target in DEFAULT_TBOX_TARGET.items()
            for index in range(target + 20)
        ]
        prefix = _weighted_tbox_order(rows, DEFAULT_TBOX_TARGET)[:300]
        counts = {
            category: sum(row["tbox_category"] == category for row in prefix)
            for category in DEFAULT_TBOX_TARGET
        }
        self.assertEqual(counts, DEFAULT_TBOX_TARGET)

    def _write_population(self, root: Path, *, tbox_count: int = 330) -> tuple[Path, Path]:
        records = []
        index = 1
        for offset in range(250):
            subtype = "TARGET_REQUIRED_CLAIM" if offset % 2 == 0 else "FORMAT_NORMALIZATION"
            records.append(_abox(f"logical-{offset}", index, "TypeA", subtype))
            index += 1
        for offset in range(400):
            records.append(_abox(f"local-{offset}", index, "TypeB", "LOCAL_TEXT_CONFIRMED"))
            index += 1
        for offset in range(320):
            records.append(_abox(f"external-{offset}", index, "TypeC", "EXTERNAL_BY_ELIMINATION"))
            index += 1
        for offset in range(tbox_count):
            records.append(_tbox(f"tbox-{offset}", index, offset))
            index += 1

        cases = root / "cases.jsonl"
        dispositions = root / "dispositions.jsonl"
        cases.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        dispositions.write_text(
            "".join(json.dumps({"case_id": record["id"], "disposition": "include"}) + "\n" for record in records),
            encoding="utf-8",
        )
        return cases, dispositions

    def test_builds_support_bank_and_frozen_populations_reproducibly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases, dispositions = self._write_population(root)
            first = root / "first"
            second = root / "second"
            summary = build_selection_artifacts(
                cases_path=cases,
                dispositions_path=dispositions,
                output_dir=first,
            )
            build_selection_artifacts(cases_path=cases, dispositions_path=dispositions, output_dir=second)

            self.assertEqual(summary["support_groups"], 32)
            self.assertEqual(summary["main_cases"], 1200)
            self.assertEqual(summary["api_cases"], 600)
            for name in ("eligibility-order.jsonl", "support-bank.json", "main-1200.json", "azure-600.json"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

            support = json.loads((first / "support-bank.json").read_text(encoding="utf-8"))
            self.assertEqual(len(support["support_sets"]["a_box_repair"]), 16)
            self.assertEqual(len(support["support_sets"]["t_box_repair"]), 16)
            self.assertEqual(len(support["support_sets"]["track_diagnosis"]), 2)
            main = json.loads((first / "main-1200.json").read_text(encoding="utf-8"))
            api = json.loads((first / "azure-600.json").read_text(encoding="utf-8"))
            self.assertEqual(main["quotas"], DEFAULT_MAIN_QUOTAS)
            self.assertEqual(api["quotas"], DEFAULT_API_QUOTAS)
            self.assertTrue(set(api["selected_case_ids"]).issubset(main["selected_case_ids"]))
            support_groups = {
                row["group_key"]
                for rows in support["support_sets"].values()
                for row in rows
            }
            self.assertTrue(support_groups.isdisjoint(main["selected_group_keys"]))

    def test_nested_expansion_preserves_parent_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases, dispositions = self._write_population(root)
            output = root / "selection"
            build_selection_artifacts(cases_path=cases, dispositions_path=dispositions, output_dir=output)
            eligibility = [
                json.loads(line)
                for line in (output / "eligibility-order.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            support = json.loads((output / "support-bank.json").read_text(encoding="utf-8"))
            parent = json.loads((output / "main-1200.json").read_text(encoding="utf-8"))
            expanded = materialize_population(
                name="main-1220",
                quotas={"IC-L": 235, "IC-G": 380, "IC-E-elim": 300, "TBOX": 305},
                eligibility_rows=eligibility,
                support_bank=support,
                parent_manifest=parent,
            )
            self.assertEqual(expanded["case_count"], 1220)
            self.assertTrue(set(parent["selected_case_ids"]).issubset(expanded["selected_case_ids"]))
            self.assertEqual(expanded["parent"], {"name": "main-1200", "nesting_proven": True})

    def test_decreasing_quota_is_not_a_nested_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases, dispositions = self._write_population(root)
            output = root / "selection"
            build_selection_artifacts(cases_path=cases, dispositions_path=dispositions, output_dir=output)
            eligibility = [
                json.loads(line)
                for line in (output / "eligibility-order.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            support = json.loads((output / "support-bank.json").read_text(encoding="utf-8"))
            parent = json.loads((output / "main-1200.json").read_text(encoding="utf-8"))
            with self.assertRaisesRegex(ValueError, "not a nested expansion"):
                materialize_population(
                    name="invalid",
                    quotas={"IC-L": 229, "IC-G": 376, "IC-E-elim": 295, "TBOX": 300},
                    eligibility_rows=eligibility,
                    support_bank=support,
                    parent_manifest=parent,
                )

    def test_tbox_underfill_is_redistributed_without_reducing_population(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases, dispositions = self._write_population(root, tbox_count=290)
            output = root / "selection"
            build_selection_artifacts(cases_path=cases, dispositions_path=dispositions, output_dir=output)
            main = json.loads((output / "main-1200.json").read_text(encoding="utf-8"))
            self.assertEqual(main["case_count"], 1200)
            self.assertLess(main["quotas"]["TBOX"], DEFAULT_MAIN_QUOTAS["TBOX"])
            self.assertEqual(sum(main["quotas"].values()), 1200)

    def test_selection_requires_complete_disposition_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases, dispositions = self._write_population(root)
            lines = dispositions.read_text(encoding="utf-8").splitlines()
            dispositions.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "coverage must exactly equal"):
                build_selection_artifacts(cases_path=cases, dispositions_path=dispositions, output_dir=root / "out")


if __name__ == "__main__":
    unittest.main()
